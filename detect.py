import face_embedding
import argparse
import cv2
import os
import sys
import datetime
import time
import pickle
import numpy as np
import mxnet as mx
import sklearn
from sklearn.cluster import DBSCAN
from essh_detector import ESSHDetector
from mtcnn_detector import MtcnnDetector
import face_preprocess


parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='./model/model-r50-gg/model,0', help='path to load model.') #0.875
# parser.add_argument('--output', default='/media/3T_disk/my_datasets/iqiyi_vid/det_trainval', help='')
parser.add_argument('--output', default='/media/3T_disk/my_datasets/iqiyi_vid/det_test', help='')
parser.add_argument('--dataset', default='/media/3T_disk/my_datasets/iqiyi_vid', help='')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='essh option')
parser.add_argument('--sampling', default=3, type=int, help='')
parser.add_argument('--split', default='', type=str, help='')
parser.add_argument('--threshold', default=0.9, type=float, help='clustering dist threshold')
parser.add_argument('--quality-threshold', default=10.0, type=float, help='quality threshold')
# parser.add_argument('--stage', default='trainval', type=str, help='choose trainval or test stage')
parser.add_argument('--stage', default='test', type=str, help='choose trainval or test stage')
args = parser.parse_args()


model = None

PARTS = [1]
MAX_LABEL = 574

PARTS = [1, 2, 3]
MAX_LABEL = 99999

print(args, MAX_LABEL, PARTS)


SPLIT = [0, 1]
if len(args.split)>0:
  _v = args.split.split(',')
  SPLIT[0] = int(_v[0])
  SPLIT[1] = int(_v[1])

print('SPLIT:', SPLIT)


detector = ESSHDetector('./model/ssh-model/essh', 0, ctx_id=args.gpu, test_mode=False)

def get_faces(video, is_train=True):
  R = []
  sampling = args.sampling
  while True:
    cap = cv2.VideoCapture(video)
    frame_num = 0
    while cap.isOpened(): 
      ret,frame = cap.read() 
      if frame is None:
        break
      frame_num+=1
      if frame_num%sampling!=0:
        continue
      frame = cv2.resize(frame, (888, 480))
      #print('frame', frame.shape)
      #faces = model.get_all(frame)
      faces = detector.detect(frame, 0.5, scales=[1.0])
      if faces is None or faces.shape[0]==0:
        continue
      det = faces
      #det = np.zeros( (len(faces), 4), dtype=np.float32)
      #for f in range(len(faces)):
      #  _face = faces[f]
      #  det[f] = _face[1]
      img_size = np.asarray(frame.shape)[0:2]
      bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
      img_center = img_size / 2
      offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
      offset_dist_squared = np.sum(np.power(offsets,2.0),0)
      bindex = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
      face = faces[bindex]
      bbox = face[0:5].reshape(1,5)
      landmark = face[5:15].reshape((5,2))
      # for b in bbox:
      #   cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))
      # for p in landmark:
      #   cv2.circle(frame, (int(p[0]), int(p[1])), 1, (0, 0, 255), 2)
      # cv2.imshow("detection result", frame)
      # cv2.waitKey(0)
      # print(bbox)
      # print(landmark)
      rimg = face_preprocess.preprocess(frame, bbox, landmark, image_size='112,112')
      R.append( (frame_num, rimg, bbox, landmark) )
      # cv2.imshow('alignment result', rimg)
      # cv2.waitKey(0)

    if is_train:
      break
    else:
      if len(R)>0 or sampling==1:
        break
    sampling = 1
  return R

def compact_faces(R):
  if len(R)==0:
    return []
  imgs = []
  for r in R:
    imgs.append(r[1])
  features = model.get_features(imgs)
  assert features.shape[0]==len(imgs)
  X = []
  R2 = []
  for i in xrange(features.shape[0]):
    f = features[i]
    norm = np.linalg.norm(f)
    if norm<args.quality_threshold:
      continue
    img = R[i][1]
    img_encode = cv2.imencode('.jpg', img)[1]
    #print(img_encode.__class__)
    # print('norm: ',norm)

    R2.append( (R[i][0], img_encode, R[i][2], R[i][3]) )
  return R2


valid_name = {}
for line in open(os.path.join(args.dataset, 'gt_v2', 'IQIYI_VID.txt'), 'r'):
  name = line.strip()
  valid_name[name] = 1

# trainval stage
if args.stage == 'trainval':
  name2path = {}
  val_names = []
  for part in PARTS:
    dataset = os.path.join(args.dataset, 'IQIYI_VID_DATA_Part%d'%part)
    for subdir in ['IQIYI_VID_TRAIN', 'IQIYI_VID_VAL']:
      _dir = os.path.join(dataset, subdir)
      _list = os.listdir(_dir)
      _list = sorted(_list)
      for video_file in _list:
        name = video_file
        if name not in valid_name:
          continue
        path = os.path.join(_dir, name)
        # assert name not in name2path
        name2path[name] = path
        if subdir=='IQIYI_VID_VAL':
          val_names.append(name)
  print(len(name2path), len(val_names))


  gt_label = {}
  gt_map = {}
  ret_map = {}
  for line in open(os.path.join(args.dataset, 'gt_v2', 'val_v2.txt'), 'r'):
    vec = line.strip().split()
    label = int(vec[0])
    if label>MAX_LABEL:
      continue
    if not label in gt_map:
      gt_map[label] = []
    if not label in ret_map:
      ret_map[label] = []
    for name in vec[1:]:
      assert name not in gt_label
      assert name in valid_name
      if name not in name2path:
        continue
      gt_label[name] = label
      gt_map[label].append(name)

  train_filename = args.output
  assert not os.path.exists(train_filename)
  f = open(train_filename, 'wb')
  if model is None:
    model = face_embedding.FaceModel(model=args.model, gpu_id = args.gpu, feature_norm=True)
  vid = 0
  for line in open(os.path.join(args.dataset, 'gt_v2', 'train_v2.txt'), 'r'):
    name, label = line.strip().split()
    label = int(label)
    if label>MAX_LABEL:
      continue
    #if name not in valid_name:
    #  continue
    if name not in name2path:
      continue
    vid+=1
    namehash = hash(name)
    mod = namehash%SPLIT[1]
    #print(namehash, mod)
    if mod!=SPLIT[0]:
      continue
    video_file = name2path[name]
    #assert os.path.exists(video_file)
    if not os.path.exists(video_file):
      print('XXXX not exists', video_file)
      continue
    timea = datetime.datetime.now()
    R = get_faces(video_file, True)
    timeb = datetime.datetime.now()
    diff = timeb - timea
    print(video_file, vid, len(R), diff.total_seconds())
    R = compact_faces(R)
    if len(R)==0:
      continue
    flag = 1
    pickle.dump((name, R, label, flag), f, protocol=pickle.HIGHEST_PROTOCOL)

  #compute val start
  vid = 0
  for name in val_names:
    label = -1
    if name in gt_label:
      label = gt_label[name]
    vid+=1
    namehash = hash(name)
    if namehash%SPLIT[1]!=SPLIT[0]:
      continue
    video_file = name2path[name]
    #assert os.path.exists(video_file)
    if not os.path.exists(video_file):
      print('XXXX not exists', video_file)
      continue
    R = get_faces(video_file, False)
    print(video_file, vid, len(R))
    R = compact_faces(R)
    if len(R)==0:
      continue
    flag = 2
    pickle.dump((name, R, label, flag), f, protocol=pickle.HIGHEST_PROTOCOL)

  f.close()

# test stage
else:
  name2path = {}
  test_names = []
  dataset = os.path.join(args.dataset, 'IQIYI_VID_TEST')
  _list = os.listdir(dataset)
  _list = sorted(_list)
  for video_file in _list:
    name = video_file
    path = os.path.join(dataset, name)
    name2path[name] = path
    test_names.append(name)
  print(len(name2path), len(test_names))

  train_filename = args.output
  assert not os.path.exists(train_filename)
  f = open(train_filename, 'wb')
  if model is None:
    model = face_embedding.FaceModel(model=args.model, gpu_id = args.gpu, feature_norm=True)

  #compute test start
  vid = 0
  for name in test_names:
    label = -1
    vid+=1
    namehash = hash(name)
    if namehash%SPLIT[1]!=SPLIT[0]:
      continue
    video_file = name2path[name]
    #assert os.path.exists(video_file)
    if not os.path.exists(video_file):
      print('XXXX not exists', video_file)
      continue
    R = get_faces(video_file, False)
    print(video_file, vid, len(R))
    R = compact_faces(R)
    if len(R)==0:
      continue
    flag = 3
    pickle.dump((name, R, label, flag), f, protocol=pickle.HIGHEST_PROTOCOL)

  f.close()

