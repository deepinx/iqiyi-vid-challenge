
import os
import pickle
import argparse
import random
import numpy as np
import sklearn
import sklearn.preprocessing

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--inputs', default='/media/3T_disk/my_datasets/iqiyi_vid/feat_trainvala', help='')
parser.add_argument('--output', default='/media/3T_disk/my_datasets/iqiyi_vid/trainvala', help='')
args = parser.parse_args()

#PARTS = [1]
#MAX_LABEL = 574

PARTS = [1, 2, 3]
MAX_LABEL = 99999
MODE = 2

print(args, MAX_LABEL, PARTS, MODE)


inputs = args.inputs.split(',')



streams = []
for input in inputs:
  filename = input
  assert os.path.exists(filename)
  f = open(filename, 'rb')
  streams.append(f)

DB = {}
DB_val = {}

DB_NAME = {}


for f in streams:
  while True:
    try:
      item = pickle.load(f)
    except:
      break
    name = item[0]
    if name not in DB_NAME:
      DB_NAME[name] = []
    DB_NAME[name].append(item)

fout = open(args.output, 'wb')
idx = 0
for name, items in DB_NAME.iteritems():
  idx+=1
  if idx%10000==0:
    print('processing', idx)
  #if len(items)!=len(streams):
  #  continue
  arrs = []
  for item in items:
    arrs.append(item[1])
  feat = np.concatenate(arrs, axis=1)
  #feat = sklearn.preprocessing.normalize(feat)
  label = items[0][2]
  flag = items[0][3]
  #if label<0:
  #  rnd = random.random()
  #  if rnd>=0.1:
  #    continue
  if label<0:
    label = 0
  pickle.dump((feat, label, flag), fout, protocol=pickle.HIGHEST_PROTOCOL)

for f in streams:
  f.close()

fout.close()

