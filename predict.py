
import os
import sys
import pickle
import mxnet as mx
import argparse
import numpy as np
import sklearn
import sklearn.preprocessing

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--model', default='./model/iqiyia1,40', help='')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--inputs', default='/media/3T_disk/my_datasets/iqiyi_vid/feat_testa', help='')
parser.add_argument('--output', default='/media/3T_disk/my_datasets/iqiyi_vid/pred_testa', help='')
args = parser.parse_args()


print(args)
MODE = 2
emb_size = 0
model = None



def get_score(feat):
  data = mx.nd.array(feat)
  db = mx.io.DataBatch(data=(data,))
  model.forward(db, is_train=False)
  xscore = model.get_outputs()[0].asnumpy()
  #print(xscore.shape)
  return xscore

inputs = args.inputs.split(',')


streams = []
for input in inputs:
  filename = input
  assert os.path.exists(filename)
  f = open(filename, 'rb')
  streams.append(f)

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

for f in streams:
  f.close()


print('total', len(DB_NAME))
fout = open(args.output, 'wb')
#ret_map = {}
batch_size = 32
pp = 0
TOPK = 100
N = 200
S = 10000.0

def process(datas):
  name_list = []
  feat_list = []
  for data in datas:
    name_list.append(data[0])
    feat_list.append(data[1])
  feats = np.array(feat_list)
  xscores = get_score(feats)
  assert len(name_list)==xscores.shape[0]
  #print(feats.shape, xscores.shape)
  for i in range(len(name_list)):
    name = name_list[i]
    xscore = xscores[i]
    pickle.dump((name, xscore), fout, protocol=pickle.HIGHEST_PROTOCOL)


#S = 1.0
buf = []
for name, items in DB_NAME.iteritems():
  pp+=1
  if pp%1000==0:
    print('processing', pp)
  #if len(items)!=len(streams):
  #  continue
  arrs = []
  for item in items:
    arrs.append(item[1])
  feat = np.concatenate(arrs, axis=1).flatten()
  if model is None:
    emb_size = len(feat)
    print('emb_size', emb_size)
    _vec = args.model.split(',')
    assert len(_vec)==2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading',prefix, epoch)
    ctx = mx.gpu(args.gpu)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers['fc7_output']
    sym = mx.sym.SoftmaxActivation(sym)
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(data_shapes=[('data', (1, emb_size))])
    model.set_params(arg_params, aux_params)
  #feat = sklearn.preprocessing.normalize(feat)
  #label = items[0][2]
  flag = items[0][3]
  # assert flag==3
  buf.append( (name, feat) )
  if len(buf)==batch_size:
    process(buf)
    buf = []

if len(buf)>0:
  process(buf)

fout.close()
sys.exit(0)


