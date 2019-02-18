from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import pickle
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
import sklearn
#sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
#import center_loss


logger = logging.getLogger()
logger.setLevel(logging.INFO)


args = None


class AccMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(AccMetric, self).__init__(
        'acc', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels, preds):
    self.count+=1
    preds = [preds[0]] #use softmax output
    for label, pred_label in zip(labels, preds):
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy()
        if label.ndim==2:
          label = label[:,0]
        label = label.astype('int32').flatten()
        assert label.shape==pred_label.shape
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)

class LossValueMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(LossValueMetric, self).__init__(
        'lossvalue', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []

  def update(self, labels, preds):
    loss = preds[-1].asnumpy()[0]
    self.sum_metric += loss
    self.num_inst += 1.0
    gt_label = preds[-2].asnumpy()
    #print(gt_label)

def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--data', default='/media/3T_disk/my_datasets/iqiyi_vid/gt_v2/trainvala', help='')
  parser.add_argument('--prefix', default='./model/iqiyia1', help='directory to save model.')
  parser.add_argument('--pretrained', default='', help='pretrained model to load')
  parser.add_argument('--ckpt', type=int, default=1, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
  parser.add_argument('--loss-type', type=int, default=4, help='loss type')
  parser.add_argument('--num-filter', type=int, default=1024, help='')
  parser.add_argument('--num-classes', type=int, default=4935, help='')
  parser.add_argument('--split', type=int, default=1, help='')
  parser.add_argument('--verbose', type=int, default=2000, help='do verification testing and model saving every verbose batches')
  parser.add_argument('--max-steps', type=int, default=0, help='max training batches')
  parser.add_argument('--begin-epoch', type=int, default=0, help='training epoch size.')
  parser.add_argument('--end-epoch', type=int, default=50, help='training epoch size.')
  parser.add_argument('--lr-step', type=str, default='30,35,40', help='steps of lr changing')
  #parser.add_argument('--lr-step', type=str, default='20,30,40,45', help='steps of lr changing')
  parser.add_argument('--network', default='r50', help='specify network')
  parser.add_argument('--lr', type=float, default=0.2, help='start learning rate')
  parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
  parser.add_argument('--mom', type=float, default=0.9, help='momentum')
  parser.add_argument('--per-batch-size', type=int, default=10, help='batch size in each context, default 4096')
  parser.add_argument('--ce-loss', default=False, action='store_true', help='if output ce loss')
  parser.add_argument('--fbn', default=False, action='store_true', help='use fixed gamma bn in last')
  args = parser.parse_args()
  return args

def load_data(args):
  data_train = []
  label_train = []
  data_val = []
  label_val = []
  emb_size = 0
  C = 5
  assert args.split<C
  f = open(args.data, 'rb')
  label_stat = [-99, -99, 0]
  while True:
    try:
      item = pickle.load(f)
    except:
      break
    feat = item[0].flatten()
    emb_size = len(feat)
    label = item[1]
    assert label<args.num_classes
    if args.split<0 and label==0:
      continue
    if label_stat[0]<-1:
      label_stat[0] = label
      label_stat[1] = label
    else:
      label_stat[0] = min(label_stat[0], label)
      label_stat[1] = max(label_stat[1], label)
    flag = item[2]
    if label==0:
      label_stat[2]+=1
    is_train = True
    if args.split==-1:
      if flag==1:
        is_train = True
      else:
        is_train = False
    elif args.split==-2:
      is_train = True
    elif args.split==0:
      rnd = random.random()
      if rnd<0.01:
        is_train = False
      else:
        is_train = True
    if is_train:
      data_train.append(feat)
      label_train.append(label)
    else:
      data_val.append(feat)
      label_val.append(label)
  f.close()
  print('label_stat', label_stat)
  #if len(data_val)==0:
  #  assert args.split>=0
  #  c = len(data_train)//C
  #  train_idx = []
  #  val_idx = []
  #  a = 0
  #  b = c*args.split
  #  train_idx.append(range(a, b))
  #  a = c*(args.split)
  #  b = c*(args.split+1)
  #  val_idx.append(range(a, b))
  #  a = c*(args.split+1)
  #  b = C
  #  train_idx.append(range(a, b))
  #  data_val = data_train[val_idx]
  #  label_val = label_train[val_idx]
  #  data_train = data_train[train_idx]
  #  label_train = label_train[train_idx]
  data_train = np.array(data_train)
  data_val = np.array(data_val)
  label_train = np.array(label_train)
  label_val = np.array(label_val)
  print('train', data_train.shape, label_train.shape)
  print('val', data_val.shape, label_val.shape)
  train_iter = mx.io.NDArrayIter(data_train, label_train, args.batch_size, True, last_batch_handle='pad')
  if data_val.shape[0]>0:
    val_iter = mx.io.NDArrayIter(data_val, label_val, args.batch_size, False, last_batch_handle='pad')
  else:
    val_iter = None
  return train_iter, val_iter, emb_size, len(label_train)

def get_symbol(args, arg_params, aux_params):
  SE = False
  num_layers = 3
  data = mx.symbol.Variable('data')
  label = mx.symbol.Variable('softmax_label')

  n_input = args.emb_size
  body = data
  for l in range(num_layers):
    shortcut = body
    n_output = args.num_filter if l<num_layers-1 else args.num_classes
    _weight = mx.symbol.Variable("stage%d_weight"%l, shape=(n_output, n_input))
    _bias = mx.symbol.Variable('stage%d_bias'%l, lr_mult=2.0, wd_mult=0.0)
    #_bias = mx.symbol.Variable('stage%d_bias'%l, lr_mult=1.0, wd_mult=1.0)
    body = mx.sym.FullyConnected(data=body, weight = _weight, bias = _bias, num_hidden=n_output, name='stage%d'%l)
    if l<num_layers-1:
      body = mx.sym.BatchNorm(data=body, name='stage%d_bn' %(l), fix_gamma=False,momentum=0.9)    
      #body = mx.sym.Activation(body, act_type='relu')
      body = mx.sym.LeakyReLU(data = body, act_type='prelu')
      if SE:
        se = mx.sym.FullyConnected(data=body, num_hidden=n_output//2, name='stage%d_se'%l)
        se = mx.sym.Activation(se, act_type='relu')
        se = mx.sym.FullyConnected(data=se, num_hidden=n_output, name='stage%d_se2'%l)
        se = mx.symbol.Activation(data=se, act_type='sigmoid', name="stage%d_se2_sigmoid"%l)
        body = mx.symbol.broadcast_mul(body, se)
      #body = mx.sym.Dropout(body, 0.2)
      if n_output==n_input:
        body = shortcut+body
    #if l<num_layers-1:
    #  body = mx.sym.Dropout(body, 0.5)
    n_input = n_output
  fc7 = body
  if args.fbn:
    fc7 = mx.sym.BatchNorm(data=fc7, name='fc7', fix_gamma=True, momentum=0.9)    
  else:
    fc7 = mx.sym.identity(data=fc7, name='fc7')
  #gt_one_hot = mx.sym.one_hot(label, depth = args.num_classes, on_value = s_m, off_value = 0.0)
  softmax = mx.symbol.SoftmaxOutput(data=fc7, label = label, name='softmax', normalization='valid')
  out_list = []
  out_list.append(softmax)
  if args.ce_loss:
    #ce_loss = mx.symbol.softmax_cross_entropy(data=fc7, label = gt_label, name='ce_loss')/args.per_batch_size
    body = mx.symbol.SoftmaxActivation(data=fc7)
    body = mx.symbol.log(body)
    _label = mx.sym.one_hot(gt_label, depth = args.num_classes, on_value = -1.0, off_value = 0.0)
    body = body*_label
    ce_loss = mx.symbol.sum(body)/args.per_batch_size
    out_list.append(mx.symbol.BlockGrad(ce_loss))
  out = mx.symbol.Group(out_list)
  return (out, arg_params, aux_params)

def train_net(args):
    ctx = []
    cvd = '0'   # os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
      for i in xrange(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
    if len(ctx)==0:
      ctx = [mx.cpu()]
      print('use cpu')
    else:
      print('gpu num:', len(ctx))
    prefix = args.prefix
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)
    end_epoch = args.end_epoch
    args.ctx_num = len(ctx)
    args.num_layers = int(args.network[1:])
    print('num_layers', args.num_layers)
    if args.per_batch_size==0:
      args.per_batch_size = 128
    args.batch_size = args.per_batch_size*args.ctx_num
    args.rescale_threshold = 0

    print('num_classes', args.num_classes)


    print('Called with argument:', args)
    train_dataiter, val_dataiter, emb_size, data_size = load_data(args)
    args.emb_size = emb_size
    print('emb_size', emb_size)
    print('data_size', data_size)

    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    arg_params = None
    aux_params = None
    sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)

    #label_name = 'softmax_label'
    #label_shape = (args.batch_size,)
    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
    )


    metric1 = AccMetric()
    eval_metrics = [mx.metric.create(metric1)]
    if args.ce_loss:
      metric2 = LossValueMetric()
      eval_metrics.append( mx.metric.create(metric2) )

    if args.network[0]=='r' or args.network[0]=='y':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    elif args.network[0]=='i' or args.network[0]=='x':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) #inception
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0/args.ctx_num
    opt = optimizer.SGD(learning_rate=base_lr, momentum=args.mom, wd=args.wd, rescale_grad=_rescale)
    #opt = optimizer.Nadam(learning_rate=base_lr, wd=args.wd, rescale_grad=_rescale, clip_gradient=5.0)
    #opt = optimizer.Adam(learning_rate=base_lr, wd=args.wd, rescale_grad=_rescale)
    som = 20
    _cb = mx.callback.Speedometer(args.batch_size, som)
    lr_step = [int(x) for x in args.lr_step.split(',')]
    begin_epoch = args.begin_epoch
    epoches = lr_step
    end_epoch = epoches[-1]
    lr_factor = 0.1
    #lr_epoch = [int(epoch) for epoch in lr_step.split(',')]
    #lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    #lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    #lr_iters = [int(epoch * len(roidb) / input_batch_size) for epoch in lr_epoch_diff]

    #lr_iters = [36000,42000] #TODO
    #lr_iters = [40000,50000,60000] #TODO
    #lr_iters = [40,50,60] #TODO
    #logger.info('lr %f lr_epoch_diff %s lr_iters %s' % (lr, lr_epoch_diff, lr_iters))
    #lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)


    highest_acc = [0.0, 0.0]  #lfw and target
    #for i in xrange(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    #lr_steps = [6000,10000,12000]
    lr_steps = []
    for ep in epoches:
      lr_steps.append(data_size*ep//args.batch_size)
    print('lr_steps', lr_steps)
    def _batch_callback(param):
      #global global_step
      global_step[0]+=1
      mbatch = global_step[0]
      for _lr in lr_steps:
        if mbatch==_lr:
          opt.lr *= 0.1
          print('lr change to', opt.lr)
          break

      _cb(param)
      if mbatch%1000==0:
        print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)

    epoch_cb = mx.callback.do_checkpoint(args.prefix, period=end_epoch)
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)

    model.fit(train_dataiter,
        begin_epoch        = begin_epoch,
        num_epoch          = end_epoch,
        eval_data          = val_dataiter,
        eval_metric        = eval_metrics,
        kvstore            = 'device',
        optimizer          = opt,
        #optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        allow_missing      = True,
        batch_end_callback = _batch_callback,
        epoch_end_callback = epoch_cb )

def main():
    #time.sleep(3600*6.5)
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

