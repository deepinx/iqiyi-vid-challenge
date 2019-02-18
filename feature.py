import face_embedding
import argparse
import cv2
import os
import sys
import datetime
import time
import pickle
import numpy as np
import sklearn
from sklearn.cluster import DBSCAN
from ssh_detector import SSHDetector
import face_preprocess

def IOU(Reframe,GTframe):
  x1 = Reframe[0]
  y1 = Reframe[1]
  width1 = Reframe[2]-Reframe[0]
  height1 = Reframe[3]-Reframe[1]
  x2 = GTframe[0]
  y2 = GTframe[1]
  width2 = GTframe[2]-GTframe[0]
  height2 = GTframe[3]-GTframe[1]
  endx = max(x1+width1,x2+width2)
  startx = min(x1,x2)
  width = width1+width2-(endx-startx)

  endy = max(y1+height1,y2+height2)
  starty = min(y1,y2)
  height = height1+height2-(endy-starty)

  if width <=0 or height <= 0:
    ratio = 0
  else:
    Area = width*height
    Area1 = width1*height1
    Area2 = width2*height2
    ratio = Area*1./(Area1+Area2-Area)
  return ratio

def brightness_aug(src, x):
  alpha = 1.0 + x
  src *= alpha
  return src

def contrast_aug(src, x):
  alpha = 1.0 + x
  coef = np.array([[[0.299, 0.587, 0.114]]])
  gray = src * coef
  gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
  src *= alpha
  src += gray
  return src

def saturation_aug(src, x):
  alpha = 1.0 + x
  coef = np.array([[[0.299, 0.587, 0.114]]])
  gray = src * coef
  gray = np.sum(gray, axis=2, keepdims=True)
  gray *= (1.0 - alpha)
  src *= alpha
  src += gray
  return src

def color_aug(img, x):
  augs = [brightness_aug, contrast_aug, saturation_aug]
  for aug in augs:
    img = aug(img, x)
  return img

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='./model/model-r100-gg/model,0', help='path to load model.') #0.875
parser.add_argument('--input', default='/media/3T_disk/my_datasets/iqiyi_vid/gt_v2/det_trainval', help='')
parser.add_argument('--output', default='/media/3T_disk/my_datasets/iqiyi_vid/gt_v2/feata', help='')
parser.add_argument('--dataset', default='/gpu/data1/jiaguo/iqiyi', help='')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 2 means using R+O, else using O')
parser.add_argument('--sampling', default=3, type=int, help='')
parser.add_argument('--aug', default=0, type=int, help='')
parser.add_argument('--threshold', default=0.9, type=float, help='clustering dist threshold')
args = parser.parse_args()


model = None

#PARTS = [1]
#MAX_LABEL = 574

PARTS = [1, 2, 3]
MAX_LABEL = 99999

print(args, MAX_LABEL, PARTS)

def get_feature(R, label, flag):
  if len(R)==0:
    return None
  R2 = []
  for r in R:
    if r[0]%args.sampling!=0:
      continue
    R2.append(r)
  R = R2
  if len(R)==0:
    return None
  imgs = []
  for r in R:
    img = cv2.imdecode(r[1], cv2.IMREAD_COLOR)
    imgs.append(img)
  c = len(imgs)
  if args.aug>0:
    _c = len(imgs)
    for i in range(_c):
      img = imgs[i]
      fimg = cv2.flip(img, 1)
      imgs.append(fimg)

  if args.aug>1:
    _c = len(imgs)
    for i in range(_c):
      imgs[i] = imgs[i].astype(np.float32)
    for i in range(_c):
      img = imgs[i]
      fimg = color_aug(img, -0.1)
      imgs.append(fimg)
    for i in range(_c):
      img = imgs[i]
      fimg = color_aug(img, 0.1)
      imgs.append(fimg)

  _features = model.get_features(imgs)
  assert _features.shape[0]==len(imgs)
  assert _features.shape[0]%c==0
  ic = 0
  features = None
  while ic<c:
    feat = _features[ic:ic+c,:]
    if features is None:
      features = feat
    else:
      features += feat
    ic+=c
  X = []
  poses = []
  for i in xrange(features.shape[0]):
    f = features[i]
    norm = np.linalg.norm(f)
    f /= norm
    X.append(f)
    pose = 0
    #pose, _, _, _, _ = SSHDetector.check_large_pose(R[i][3], R[i][2])
    #if flag==2: #is val
    #  pose, _, _, _, _ = SSHDetector.check_large_pose(R[i][3], R[i][2])

    poses.append(pose)
  X = np.array(X)
  pose_map = {}
  pose_keys = []
  for i in range(len(poses)):
    pose = poses[i]
    if not pose in pose_map:
      pose_map[pose] = []
      pose_keys.append(pose)
    pose_map[pose].append(i)
  pose_keys = sorted(pose_keys)
  F = []
  for pose in pose_keys:
    #if pose!=0:
    #  continue
    arr = pose_map[pose]
    x = X[arr,:]
    feat = np.sum(x, axis=0)
    norm = np.linalg.norm(feat)
    feat /= norm
    F.append(feat)

  if len(F)==0:
    return None
  feat = np.array(F)
  return feat


train_filename = args.output
assert not os.path.exists(train_filename)
f = open(train_filename, 'wb')
if model is None:
  model = face_embedding.FaceModel(model=args.model, gpu_id = args.gpu, feature_norm=True)

vid = 0
fin = open(args.input, 'rb')
while True:
  try:
    item = pickle.load(fin)
  except:
    break
  vid+=1
  name = item[0]
  R = item[1]
  label = item[2]
  flag = item[3]
  print(name, 'label', label, 'lenR', len(R), 'flag', flag, 'vid', vid)
  feat = get_feature(R, label, flag)
  if feat is None:
    continue
  pickle.dump((name, feat, label, flag), f, protocol=pickle.HIGHEST_PROTOCOL)

fin.close()
f.close()


