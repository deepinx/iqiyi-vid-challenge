
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
parser.add_argument('--model', default='', help='')
parser.add_argument('--gpu', default=6, type=int, help='gpu id')
parser.add_argument('--inputs', default='', help='')
parser.add_argument('--output', default='', help='')
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
  assert flag==3
  buf.append( (name, feat) )
  if len(buf)==batch_size:
    process(buf)
    buf = []

  #DB_test[name] = feat
  #print(feat.shape)
  #index = np.argsort(xscore)[::-1]
  #index = index[:N]
  #idfound = False
  #idx = -1
  #for im in index:
  #  idx+=1
  #  label = im
  #  if label==0:
  #    #if S==0 and idx==0:
  #    #  break
  #    continue
  #  score = xscore[im]
  #  #if idfound:
  #  if idx>0:
  #    score /= S
  #  if label not in ret_map:
  #    ret_map[label] = []
  #  ret_map[label].append( (name, score) )
  #  idfound = True

if len(buf)>0:
  process(buf)

fout.close()
sys.exit(0)

out_filename='./submita.txt'
outf = open(out_filename, 'w')
out_filename2='./submita_score.txt'
outf2 = open(out_filename2, 'w')
empty_count=0
min_len = 99999
for label, ret_list in ret_map.iteritems():
  ret_list = sorted(ret_list, key = lambda x : x[1], reverse=True)
  if TOPK>0 and len(ret_list)>TOPK:
    ret_list = ret_list[:TOPK]
  min_len = min(min_len, len(ret_list))
  out_items = [str(label)]
  out_items2 = [str(label)]
  for ir, r in enumerate(ret_list):
    name = r[0]
    score = r[1]
    out_items.append(name)
    out_items2.append('%.3f'%score)
  outf.write("%s\n"%(' '.join(out_items)))
  outf2.write("%s\n"%(' '.join(out_items2)))
outf.close()
outf2.close()
print('min', min_len)

