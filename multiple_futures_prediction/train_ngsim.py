#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019-2020 Apple Inc. All Rights Reserved.
#

from typing import List, Set, Dict, Tuple, Optional, Union, Any
import numpy as np
np.set_printoptions(suppress=1)
import time
import math
import glob
from attrdict import AttrDict
import gin
import torch
from torch.utils.data import DataLoader
from multiple_futures_prediction.dataset_ngsim import *
from multiple_futures_prediction.model_ngsim import mfpNet
from multiple_futures_prediction.my_utils import *

def eval(metric: str, net: torch.nn.Module, params: AttrDict, data_loader: DataLoader, bStepByStep: bool, 
        use_forcing: int, y_mean: np.ndarray, num_batches: int, dataset_name: str) -> torch.Tensor:
  """Evaluation function for validation and test data.  
  
  Given a MFP network, data loader, evaulate either NLL or RMSE error.
  """  
  print('eval ', dataset_name)
  num = params.fut_len_orig_hz//params.subsampling
  lossVals = torch.zeros(num)
  counts = torch.zeros(num)

  for i, data in enumerate(data_loader):
    if i >= num_batches:
      break      
    hist, nbrs, mask, fut, mask, context, nbrs_info = data
    if params.use_cuda:
      hist = hist.cuda()
      nbrs = nbrs.cuda()
      mask = mask.cuda()            
      fut = fut.cuda()
      mask = mask.cuda()
      if context is not None:
        context = context.cuda()

    if metric == 'nll':      
      fut_preds, modes_pred = net.forward_mfp(hist, nbrs, mask, context, nbrs_info, fut, bStepByStep, use_forcing=use_forcing)
      if params.modes == 1:
        if params.remove_y_mean:
          fut_preds[0][:,:,:2] += y_mean.unsqueeze(1).to(fut.device)
        l, c = nll_loss_test(fut_preds[0], fut, mask)
      else:
        l, c = nll_loss_test_multimodes(fut_preds, fut, mask, modes_pred, y_mean.to(fut.device) )                        
    else: # RMSE error
      assert params.modes == 1
      fut_preds, modes_pred = net.forward_mfp(hist, nbrs, mask, context, nbrs_info, fut, bStepByStep, use_forcing=use_forcing)
      if params.modes == 1:
        if params.remove_y_mean:
          fut_preds[0][:,:,:2] += y_mean.unsqueeze(1).to(fut.device)
        l, c = mse_loss_test(fut_preds[0], fut, mask)

    lossVals  += l.detach().cpu()
    counts    += c.detach().cpu()

  if metric == 'nll':
    err = lossVals / counts
    print(lossVals / counts)
  else:
    err = torch.pow(lossVals / counts,0.5)*0.3048
    print( err ) # Calculate RMSE and convert from feet to meters
  return err

def get_mean( train_data_loader: DataLoader, batches: Optional[int]=200 ) -> np.ndarray:
  """Compute the means over some samples from the training data."""  
  yy = []
  counters = None
  for i, data in enumerate(train_data_loader):        
    if i > batches: # type: ignore
      break
    hist, nbrs, _, fut, fut_mask, _, _ = data
    target = fut.cpu().numpy()
    valid = fut_mask.cpu().numpy().sum(axis=1)

    if counters is None:
      counters = np.zeros_like( valid )
    counters += valid

    isinvalid = (fut_mask == 0)        
    target[isinvalid] = 0
    yy.append( target )
  
  Y = np.concatenate(yy, axis=1)
  y_mean= np.divide( np.sum(Y,axis=1), counters)
  return y_mean
    
def setup_logger(root_dir: str, SCENARIO_NAME: str ) -> Tuple[Any, Any]:
  """Setup the data logger for logging."""
  import glob
  from subprocess import call
  import time
  import datetime
  import os

  timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y.%m.%d_%H.%M.%S')      
  logging_dir =  root_dir+"%s_%s/"%(SCENARIO_NAME, timestamp)
  if not os.path.isdir(logging_dir):
    os.makedirs(logging_dir)
    os.makedirs(logging_dir+'/checkpoints')    
    print ("! " + logging_dir + " CREATED!")
  
  logger_file = open(logging_dir+'/log', 'w')
  return logger_file, logging_dir

################################################################################
@gin.configurable
class Params(object):
  def __init__(self, log:bool=False,  # save checkpoints? 
        modes:int=2,                # how many latent modes
        use_cuda:bool=True,		
        encoder_size:int=16,        # encoder latent layer size
        decoder_size:int=16,        # decoder latent layer size        
        subsampling:int=2,          # factor subsample in time 
        hist_len_orig_hz:int=30,    # number of original history samples
        fut_len_orig_hz:int=50,     # number of original future samples
        dyn_embedding_size:int=32,  # dynamic embedding size
        input_embedding_size:int=32,        # input embedding size
        dec_nbr_enc_size:int=8,             # decoder neighbors encode size
        nbr_atten_embedding_size:int=80,    # neighborhood attention embedding size
        seed:int=1234,  
        remove_y_mean:bool=False,    # normalize by remove mean of the future trajectory
        use_gru:bool=True,           # GRUs instead of LSTMs
        bi_direc:bool=False,         # bidrectional
        self_norm:bool=False,        # normalize with respect to the current time 
        data_aug:bool=False,         # data augment
        use_context:bool=False,      # use contextual image as additional input
        nll:bool=True,               # negative log-liklihood loss
        use_forcing:int=0,          # teacher forcing
        iter_per_err:int=100,       # iterations to display errors
        iter_per_eval:int=1000,     # iterations to eval on validation set 
        pre_train_num_updates:int=200000,   # how many iterations for pretraining
        updates_div_by_10:int=100000,       # at what iteration to divide the learning rate by 10.0
        nbr_search_depth:int=10,            # how deep do we search for neighbors
        lr_init:float=0.001,                  # initial learning rate        
        min_lr:float=0.00005,                 # minimal learning rate
        iters_per_save:int=1500 ) -> None :         
        self.params = AttrDict(locals())
  def __call__(self) -> Any:
    return self.params
################################################################################

def train( params: AttrDict ) -> Any :
  """Main training function."""
  torch.manual_seed( params.seed ) #type: ignore
  np.random.seed( params.seed )

  ############################  
  batch_size = 1
  data_hz                = 10
  ns_between_samples     = (1.0/data_hz)*1e9
  d_s = params.subsampling
  t_h = params.hist_len_orig_hz
  t_f = params.fut_len_orig_hz
  NUM_WORKERS = 1
  
  DATA_PATH = 'multiple_futures_prediction/'  

  # Loading the dataset.  
  train_set = NgsimDataset(DATA_PATH + 'ngsim_data/TrainSet.mat', t_h, t_f, d_s, params.encoder_size, params.use_gru, params.self_norm,
                                params.data_aug, params.use_context, params.nbr_search_depth)
  val_set   = NgsimDataset(DATA_PATH + 'ngsim_data/ValSet.mat', t_h, t_f, d_s, params.encoder_size, params.use_gru, params.self_norm,
                                params.data_aug, params.use_context, params.nbr_search_depth)
  test_set  = NgsimDataset(DATA_PATH + 'ngsim_data/TestSet.mat', t_h, t_f, d_s, params.encoder_size, params.use_gru, params.self_norm,
                                params.data_aug, params.use_context, params.nbr_search_depth)
  train_data_loader = DataLoader(train_set,batch_size=batch_size, shuffle=1, num_workers=NUM_WORKERS, collate_fn=train_set.collate_fn,drop_last=True) # type: ignore
  val_data_loader   = DataLoader(val_set,batch_size=batch_size,shuffle=0, num_workers=NUM_WORKERS, collate_fn=val_set.collate_fn,drop_last=True) #type: ignore
  test_data_loader  = DataLoader(test_set,batch_size=batch_size,  shuffle=0, num_workers=NUM_WORKERS,collate_fn=test_set.collate_fn,drop_last=False) #type: ignore

  # Compute or load existing mean over future trajectories.
  if os.path.exists(DATA_PATH+'ngsim_data/y_mean.pkl'):
    y_mean = pickle.load( open(DATA_PATH+'ngsim_data/y_mean.pkl', 'rb'))    
  else:
    y_mean = get_mean(train_data_loader)
    pickle.dump( y_mean, open(DATA_PATH+'ngsim_data/y_mean.pkl', 'wb') )

  # Initialize network
  net = mfpNet( params )
  if params.use_cuda:
    net = net.cuda() #type: ignore
  
  net.y_mean = y_mean
  y_mean = torch.tensor(net.y_mean)

  if params.log:
    logger_file, logging_dir = setup_logger(DATA_PATH+"./checkpts/", 'NGSIM' )
    
  train_loss: List = []
  val_loss: List = []
  
  MODE='Pre'  # For efficiency, we first pre-train w/o interactive rollouts.
  num_updates = 0
  optimizer = None

  for epoch_num in range(20):
    if MODE == 'EndPre':
      MODE = 'Train'
      print('Training with interactive rollouts.')
      bStepByStep = True
    else:
      print('Pre-training without interactive rollouts.')
      bStepByStep = False        

    # Average losses.
    avg_tr_loss = 0.
    avg_tr_time = 0.
    loss_counter = 0.0

    for i, data in enumerate(train_data_loader):
      if num_updates > params.pre_train_num_updates and MODE == 'Pre':
        MODE = 'EndPre'
        break

      lr_fac = np.power(0.1, num_updates // params.updates_div_by_10 )
      lr = max( params.min_lr, params.lr_init*lr_fac)
      if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr) #type: ignore 
      elif lr != optimizer.defaults['lr']:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr) 
      
      st_time = time.time()
      hist, nbrs, mask, fut, mask, context, nbrs_info = data
      
      if params.remove_y_mean:
        fut = fut-y_mean.unsqueeze(1)
     
      if params.use_cuda:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        mask = mask.cuda()
        fut = fut.cuda()
        mask = mask.cuda()
        if context is not None:
          context = context.cuda()

      # Forward pass.
      fut_preds, modes_pred = net.forward_mfp(hist, nbrs, mask, context, nbrs_info, fut, bStepByStep)              
      if params.modes == 1:
        l = nll_loss(fut_preds[0], fut, mask)
      else:
        l = nll_loss_multimodes(fut_preds, fut, mask, modes_pred) # type: ignore

      # Backprop.
      optimizer.zero_grad()
      l.backward()
      torch.nn.utils.clip_grad_norm_(net.parameters(), 10)  #type: ignore    
      optimizer.step()        
      num_updates += 1

      batch_time = time.time()-st_time
      avg_tr_loss += l.item() 
      avg_tr_time += batch_time

      effective_batch_sz = float(hist.shape[1])
      if num_updates % params.iter_per_err == params.iter_per_err-1:            
        print("Epoch no:",epoch_num,"update:",num_updates, "| Avg train loss:",
                format(avg_tr_loss/100,'0.4f'), " learning_rate:%.5f"%lr)
        train_loss.append(avg_tr_loss/100)
        
        if params.log:
          msg_str_ = ("Epoch no:",epoch_num,"update:",num_updates, "| Avg train loss:",
                      format(avg_tr_loss/100,'0.4f'), " learning_rate:%.5f"%lr) 
          msg_str = str([str(ss) for ss in msg_str_]) 
          logger_file.write(msg_str+'\n') 
          logger_file.flush()

        avg_tr_loss = 0.
        if num_updates % params.iter_per_eval == params.iter_per_eval-1:
          print("Starting eval")                
          val_nll_err = eval(  'nll', net, params, val_data_loader, bStepByStep,
                               use_forcing=params.use_forcing, y_mean=y_mean, 
                               num_batches=500, dataset_name='val_dl nll')
          
          if params.log:
            logger_file.write('val nll: ' + str(val_nll_err)+'\n')
            logger_file.flush()

      # Save weights.
      if params.log and num_updates % params.iters_per_save == params.iters_per_save-1:
        msg_str = '\nSaving state, update iter:%d %s'%(num_updates, logging_dir)
        print(msg_str)
        logger_file.write( msg_str ); logger_file.flush()
        torch.save(net.state_dict(), logging_dir + '/checkpoints/ngsim_%06d'%num_updates + '.pth') #type: ignore


