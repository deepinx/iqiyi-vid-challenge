from __future__ import print_function
import sys
import os
import time
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import cv2
from rcnn.logger import logger
from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes, landmark_pred
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper
from rcnn.processing.bbox_transform import bbox_overlaps

class SSHDetector:
  def __init__(self, prefix, epoch, ctx_id=0, test_mode=False):
    self.ctx_id = ctx_id
    self.ctx = mx.gpu(self.ctx_id)
    self.fpn_keys = []
    self._feat_stride_fpn = [32, 16, 8]

    for s in self._feat_stride_fpn:
        self.fpn_keys.append('stride%s'%s)

    #self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(base_size=fpn_base_size, scales=self._scales, ratios=self._ratios)))
    self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn()))
    for k in self._anchors_fpn:
      v = self._anchors_fpn[k].astype(np.float32)
      self._anchors_fpn[k] = v

    self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
    self._rpn_pre_nms_top_n = 1000
    #self._rpn_post_nms_top_n = rpn_post_nms_top_n
    #self.score_threshold = 0.05
    self.nms_threshold = 0.3
    self._bbox_pred = nonlinear_pred
    self._landmark_pred = landmark_pred
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    self.nms = gpu_nms_wrapper(self.nms_threshold, self.ctx_id)
    self.pixel_means = np.array([103.939, 116.779, 123.68]) #BGR
    #self.pixel_means = config.PIXEL_MEANS
    print('means', self.pixel_means)
    self.use_landmarks = False
    if len(sym)//len(self._feat_stride_fpn)==3:
      self.use_landmarks = True
    print('use_landmarks', self.use_landmarks)

    if not test_mode:
      image_size = (640, 640)
      self.model = mx.mod.Module(symbol=sym, context=self.ctx, label_names = None)
      self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
      self.model.set_params(arg_params, aux_params)
    else:
      from rcnn.core.module import MutableModule
      image_size = (2400, 2400)
      data_shape = [('data', (1,3,image_size[0], image_size[1]))]
      self.model = MutableModule(symbol=sym, data_names=['data'], label_names=None,
                                context=self.ctx, max_data_shapes=data_shape)
      self.model.bind(data_shape, None, for_training=False)
      self.model.set_params(arg_params, aux_params)


  def detect(self, img, threshold=0.05, scales=[1.0]):
    proposals_list = []
    scores_list = []
    landmarks_list = []

    for im_scale in scales:

      if im_scale!=1.0:
        im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
      else:
        im = img
      im = im.astype(np.float32)
      #self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
      im_info = [im.shape[0], im.shape[1], im_scale]
      im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
      for i in range(3):
          im_tensor[0, i, :, :] = im[:, :, 2 - i] - self.pixel_means[2 - i]
      data = nd.array(im_tensor)
      db = mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])
      self.model.forward(db, is_train=False)
      net_out = self.model.get_outputs()
      pre_nms_topN = self._rpn_pre_nms_top_n
      #post_nms_topN = self._rpn_post_nms_top_n
      #min_size_dict = self._rpn_min_size_fpn

      for _idx,s in enumerate(self._feat_stride_fpn):
          if len(scales)>1 and s==32 and im_scale==scales[-1]:
            continue
          _key = 'stride%s'%s
          stride = int(s)
          if self.use_landmarks:
            idx = _idx*3
          else:
            idx = _idx*2
          #print('getting', im_scale, stride, idx, len(net_out), data.shape, file=sys.stderr)
          scores = net_out[idx].asnumpy()
          #print(scores.shape)
          #print('scores',stride, scores.shape, file=sys.stderr)
          scores = scores[:, self._num_anchors['stride%s'%s]:, :, :]

          idx+=1
          bbox_deltas = net_out[idx].asnumpy()

          #if DEBUG:
          #    print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
          #    print 'scale: {}'.format(im_info[2])

          _height, _width = int(im_info[0] / stride), int(im_info[1] / stride)
          height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

          A = self._num_anchors['stride%s'%s]
          K = height * width
          anchors_fpn = self._anchors_fpn['stride%s'%s]
          anchors = anchors_plane(height, width, stride, anchors_fpn)
          #print((height, width), (_height, _width), anchors.shape, bbox_deltas.shape, scores.shape, file=sys.stderr)
          anchors = anchors.reshape((K * A, 4))
          #print('HW', (height, width), (_height, _width), file=sys.stderr)
          #print('anchors_fpn', anchors_fpn.shape, file=sys.stderr)
          #print('anchors', anchors.shape, file=sys.stderr)
          #print('bbox_deltas', bbox_deltas.shape, file=sys.stderr)
          #print('scores', scores.shape, file=sys.stderr)


          scores = self._clip_pad(scores, (height, width))
          scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

          #print('pre', bbox_deltas.shape, height, width)
          bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
          #print('after', bbox_deltas.shape, height, width)
          bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))


          #print(anchors.shape, bbox_deltas.shape, A, K, file=sys.stderr)
          proposals = self._bbox_pred(anchors, bbox_deltas)
          proposals = clip_boxes(proposals, im_info[:2])


          #keep = self._filter_boxes(proposals, min_size_dict['stride%s'%s] * im_info[2])
          #proposals = proposals[keep, :]
          #scores = scores[keep]
          #print('333', proposals.shape)

          scores_ravel = scores.ravel()
          order = scores_ravel.argsort()[::-1]
          if pre_nms_topN > 0:
              order = order[:pre_nms_topN]
          proposals = proposals[order, :]
          scores = scores[order]

          proposals /= im_scale

          proposals_list.append(proposals)
          scores_list.append(scores)

          if self.use_landmarks:
            idx+=1
            landmark_deltas = net_out[idx].asnumpy()
            landmark_deltas = self._clip_pad(landmark_deltas, (height, width))
            landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 10))
            #print(landmark_deltas.shape, landmark_deltas)
            landmarks = self._landmark_pred(anchors, landmark_deltas)
            landmarks = landmarks[order, :]
            landmarks /= im_scale
            landmarks_list.append(landmarks)
            #proposals = np.hstack((proposals, landmarks))

    proposals = np.vstack(proposals_list)
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    #if config.TEST.SCORE_THRESH>0.0:
    #  _count = np.sum(scores_ravel>config.TEST.SCORE_THRESH)
    #  order = order[:_count]
    #if pre_nms_topN > 0:
    #    order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]
    if self.use_landmarks:
      landmarks = np.vstack(landmarks_list)
      landmarks = landmarks[order]

    det = np.hstack((proposals, scores)).astype(np.float32)
    keep = self.nms(det)
    if self.use_landmarks:
      det = np.hstack((det, landmarks))
    det = det[keep, :]
    if threshold>0.0:
      keep = np.where(det[:, 4] >= threshold)[0]
      det = det[keep, :]
    return det

  @staticmethod
  def check_large_pose(landmark, bbox):
    assert landmark.shape==(5,2)
    assert len(bbox)==4
    def get_theta(base, x, y):
      vx = x-base
      vy = y-base
      vx[1] *= -1
      vy[1] *= -1
      tx = np.arctan2(vx[1], vx[0])
      ty = np.arctan2(vy[1], vy[0])
      d = ty-tx
      d = np.degrees(d)
      #print(vx, tx, vy, ty, d)
      #if d<-1.*math.pi:
      #  d+=2*math.pi
      #elif d>math.pi:
      #  d-=2*math.pi
      if d<-180.0:
        d+=360.
      elif d>180.0:
        d-=360.0
      return d
    landmark = landmark.astype(np.float32)

    theta1 = get_theta(landmark[0], landmark[3], landmark[2])
    theta2 = get_theta(landmark[1], landmark[2], landmark[4])
    #print(va, vb, theta2)
    theta3 = get_theta(landmark[0], landmark[2], landmark[1])
    theta4 = get_theta(landmark[1], landmark[0], landmark[2])
    theta5 = get_theta(landmark[3], landmark[4], landmark[2])
    theta6 = get_theta(landmark[4], landmark[2], landmark[3])
    theta7 = get_theta(landmark[3], landmark[2], landmark[0])
    theta8 = get_theta(landmark[4], landmark[1], landmark[2])
    #print(theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8)
    left_score = 0.0
    right_score = 0.0
    up_score = 0.0
    down_score = 0.0
    if theta1<=0.0:
      left_score = 10.0
    elif theta2<=0.0:
      right_score = 10.0
    else:
      left_score = theta2/theta1
      right_score = theta1/theta2
    if theta3<=10.0 or theta4<=10.0:
      up_score = 10.0
    else:
      up_score = max(theta1/theta3, theta2/theta4)
    if theta5<=10.0 or theta6<=10.0:
      down_score = 10.0
    else:
      down_score = max(theta7/theta5, theta8/theta6)
    mleft = (landmark[0][0]+landmark[3][0])/2
    mright = (landmark[1][0]+landmark[4][0])/2
    box_center = ( (bbox[0]+bbox[2])/2,  (bbox[1]+bbox[3])/2 )
    ret = 0
    if left_score>=3.0:
      ret = 1
    if ret==0 and left_score>=2.0:
      if mright<=box_center[0]:
        ret = 1
    if ret==0 and right_score>=3.0:
      ret = 2
    if ret==0 and right_score>=2.0:
      if mleft>=box_center[0]:
        ret = 2
    if ret==0 and up_score>=2.0:
      ret = 3
    if ret==0 and down_score>=5.0:
      ret = 4
    return ret, left_score, right_score, up_score, down_score

  @staticmethod
  def _filter_boxes(boxes, min_size):
      """ Remove all boxes with any side smaller than min_size """
      ws = boxes[:, 2] - boxes[:, 0] + 1
      hs = boxes[:, 3] - boxes[:, 1] + 1
      keep = np.where((ws >= min_size) & (hs >= min_size))[0]
      return keep

  @staticmethod
  def _clip_pad(tensor, pad_shape):
      """
      Clip boxes of the pad area.
      :param tensor: [n, c, H, W]
      :param pad_shape: [h, w]
      :return: [n, c, h, w]
      """
      H, W = tensor.shape[2:]
      h, w = pad_shape

      if h < H or w < W:
        tensor = tensor[:, :, :h, :w].copy()

      return tensor

