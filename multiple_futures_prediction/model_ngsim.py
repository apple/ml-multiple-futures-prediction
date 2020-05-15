#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019-2020 Apple Inc. All Rights Reserved.
#

from typing import List, Set, Dict, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
from multiple_futures_prediction.my_utils import *

# Multiple Futures Prediction Network
class mfpNet(nn.Module):    
  def __init__(self, args: Dict) -> None:
    super(mfpNet, self).__init__() #type: ignore
    self.use_cuda = args['use_cuda'] 
    self.encoder_size = args['encoder_size']
    self.decoder_size = args['decoder_size']        
    self.out_length = args['fut_len_orig_hz']//args['subsampling']

    self.dyn_embedding_size = args['dyn_embedding_size']
    self.input_embedding_size = args['input_embedding_size']

    self.nbr_atten_embedding_size = args['nbr_atten_embedding_size']
    self.st_enc_hist_size = self.nbr_atten_embedding_size
    self.st_enc_pos_size = args['dec_nbr_enc_size'] 
    self.use_gru          = args['use_gru']
    self.bi_direc         = args['bi_direc']        
    self.use_context      = args['use_context']
    self.modes            = args['modes']
    self.use_forcing      = args['use_forcing'] # 1: Teacher forcing. 2:classmates forcing.
    
    self.hidden_fac     = 2 if args['use_gru'] else 1
    self.bi_direc_fac   = 2 if args['bi_direc'] else 1
    self.dec_fac        = 2 if args['bi_direc'] else 1   

    self.init_rbf_state_enc( in_dim=self.encoder_size*self.hidden_fac )     
    self.posi_enc_dim       = self.st_enc_pos_size
    self.posi_enc_ego_dim   = 2

    # Input embedding layer
    self.ip_emb = torch.nn.Linear(2,self.input_embedding_size) #type: ignore

    # Encoding RNN.
    if not self.use_gru:            
      self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1) # type: ignore
    else:
      self.num_layers=2
      self.enc_lstm = torch.nn.GRU(self.input_embedding_size,self.encoder_size,    # type: ignore 
                                   num_layers=self.num_layers, bidirectional=False) 

    # Dynamics embeddings.
    self.dyn_emb = torch.nn.Linear(self.encoder_size*self.hidden_fac, self.dyn_embedding_size) #type: ignore

    context_feat_size = 64 if self.use_context else 0
    self.dec_lstm = []
    self.op = []
    for k in range(self.modes):            
      if not self.use_gru:
        self.dec_lstm.append( torch.nn.LSTM(self.nbr_atten_embedding_size + self.dyn_embedding_size + #type: ignore
                                             context_feat_size+self.posi_enc_dim+self.posi_enc_ego_dim, self.decoder_size) )
      else:
        self.num_layers=2
        self.dec_lstm.append( torch.nn.GRU(self.nbr_atten_embedding_size + self.dyn_embedding_size + context_feat_size+self.posi_enc_dim+self.posi_enc_ego_dim, # type: ignore 
                                            self.decoder_size, num_layers=self.num_layers, bidirectional=self.bi_direc ))
      
      self.op.append( torch.nn.Linear(self.decoder_size*self.dec_fac, 5) ) #type: ignore
      
      self.op[k] = self.op[k]
      self.dec_lstm[k] = self.dec_lstm[k]

    self.dec_lstm   = torch.nn.ModuleList(self.dec_lstm) # type: ignore 
    self.op         = torch.nn.ModuleList(self.op )      # type: ignore

    self.op_modes = torch.nn.Linear(self.nbr_atten_embedding_size + self.dyn_embedding_size + context_feat_size, self.modes) #type: ignore

    # Nonlinear activations.
    self.leaky_relu = torch.nn.LeakyReLU(0.1) #type: ignore
    self.relu = torch.nn.ReLU() #type: ignore
    self.softmax = torch.nn.Softmax(dim=1) #type: ignore

    if self.use_context:          
      self.context_conv       = torch.nn.Conv2d(3, 16, kernel_size=3, stride=2)  #type: ignore
      self.context_conv2      = torch.nn.Conv2d(16, 16, kernel_size=3, stride=2) #type: ignore
      self.context_maxpool    = torch.nn.MaxPool2d(kernel_size=(4,2))            #type: ignore
      self.context_conv3      = torch.nn.Conv2d(16, 16, kernel_size=3, stride=2) #type: ignore
      self.context_fc         = torch.nn.Linear(16*20*3, context_feat_size)      #type: ignore
  
  def init_rbf_state_enc(self, in_dim: int ) -> None:
    """Initialize the dynamic attentional RBF encoder.
    Args:
      in_dim is the input dim of the observation.
    """
    self.sec_in_dim = in_dim
    self.extra_pos_dim = 2

    self.sec_in_pos_dim     = 2
    self.sec_key_dim        = 8
    self.sec_key_hidden_dim = 32

    self.sec_hidden_dim     = 32
    self.scale = 1.0
    self.slot_key_scale = 1.0
    self.num_slots = 8
    self.slot_keys = []
    
    # Network for computing the 'key'
    self.sec_key_net = torch.nn.Sequential( #type: ignore
                          torch.nn.Linear(self.sec_in_dim+self.extra_pos_dim, self.sec_key_hidden_dim),
                          torch.nn.ReLU(),
                          torch.nn.Linear(self.sec_key_hidden_dim, self.sec_key_dim)
                       )

    for ss in range(self.num_slots):
      self.slot_keys.append( torch.nn.Parameter( self.slot_key_scale*torch.randn( self.sec_key_dim, 1, dtype=torch.float32) ) ) #type: ignore
    self.slot_keys = torch.nn.ParameterList( self.slot_keys )  # type: ignore

    # Network for encoding a scene-level contextual feature.
    self.sec_hist_net   = torch.nn.Sequential( #type: ignore
                          torch.nn.Linear(self.sec_in_dim*self.num_slots, self.sec_hidden_dim),
                          torch.nn.ReLU(),
                          torch.nn.Linear(self.sec_hidden_dim, self.sec_hidden_dim),
                          torch.nn.ReLU(),
                          torch.nn.Linear(self.sec_hidden_dim, self.st_enc_hist_size)
                        )

    # Encoder position of other's into a feature network, input should be normalized to ref_pos.
    self.sec_pos_net = torch.nn.Sequential( #type: ignore
                          torch.nn.Linear(self.sec_in_pos_dim*self.num_slots, self.sec_hidden_dim),
                          torch.nn.ReLU(),
                          torch.nn.Linear(self.sec_hidden_dim, self.sec_hidden_dim),
                          torch.nn.ReLU(),
                          torch.nn.Linear(self.sec_hidden_dim, self.st_enc_pos_size)
                        )

  def rbf_state_enc_get_attens(self, nbrs_enc: torch.Tensor, ref_pos: torch.Tensor, nbrs_info_this: List ) -> List[torch.Tensor]:
    """Computing the attention over other agents.
    Args:
      nbrs_info_this is a list of list of (nbr_batch_ind, nbr_id, nbr_ctx_ind)
    Returns:
      attention weights over the neighbors.
    """
    assert len(nbrs_info_this) == ref_pos.shape[0]        
    if self.extra_pos_dim > 0:
      pos_enc = torch.zeros(nbrs_enc.shape[0],2, device=nbrs_enc.device)
      counter = 0
      for n in range(len(nbrs_info_this)):
        for nbr in nbrs_info_this[n]:
          pos_enc[counter,:] = ref_pos[nbr[0],:] - ref_pos[n,:]
          counter += 1          
      Key = self.sec_key_net( torch.cat( (nbrs_enc,pos_enc),dim=1) )  
      # e.g. num_agents by self.sec_key_dim
    else:
      Key = self.sec_key_net( nbrs_enc )  # e.g. num_agents by self.sec_key_dim

    attens0 = []        
    for slot in self.slot_keys:            
      attens0.append( torch.exp( -self.scale*(Key-torch.t(slot)).norm(dim=1)) )

    Atten = torch.stack(attens0, dim=0) # e.g. num_keys x num_agents
    attens = []
    counter = 0
    for n in range(len(nbrs_info_this)):
      list_of_nbrs = nbrs_info_this[n]
      counter2 = counter+len(list_of_nbrs)
      attens.append( Atten[:, counter:counter2 ] )        
      counter = counter2
    return attens

  def rbf_state_enc_hist_fwd(self, attens: List, nbrs_enc: torch.Tensor, nbrs_info_this: List) -> torch.Tensor:
    """Computes dynamic state encoding.    
    Computes dynica state encoding with precomputed attention tensor and the 
    RNN based encoding.
    Args:    
      attens is a list of [ [slots x num_neighbors]]
      nbrs_enc is num_agents by input_dim      
    Returns:
      feature vector
    """    
    out = []
    counter = 0
    for n in range(len(nbrs_info_this)):
      list_of_nbrs = nbrs_info_this[n]        
      if len(list_of_nbrs) > 0:        
        counter2 = counter+len(list_of_nbrs)
        nbr_feat = nbrs_enc[counter:counter2,:]
        out.append( torch.mm( attens[n], nbr_feat ) )
        counter = counter2
      else:
        out.append( torch.zeros(self.num_slots, nbrs_enc.shape[1] ).to(nbrs_enc.device) )  
        # if no neighbors found, use all zeros.
    st_enc = torch.stack(out, dim=0).view(len(out),-1) # num_agents by slots*enc dim
    return self.sec_hist_net(st_enc)

  def rbf_state_enc_pos_fwd(self, attens: List, ref_pos: torch.Tensor, fut_t: torch.Tensor, flatten_inds: torch.Tensor, chunks: List) -> torch.Tensor:
    """Computes the features from dynamic attention for interactive rollouts.    
    Args:    
      attens is a list of [ [slots x num_neighbors]]
      ref_pos should be (num_agents by 2)
    Returns:
      feature vector
    """
    fut = fut_t + ref_pos  #convert to 'global' frame    
    nbr_feat = torch.index_select( fut, 0, flatten_inds)
    splits = torch.split(nbr_feat, chunks, dim=0) #type: ignore
    out = []
    for n, nbr_feat in enumerate(splits):
      out.append( torch.mm( attens[n], nbr_feat - ref_pos[n,:] ) )
    pos_enc = torch.stack(out, dim=0).view(len(attens),-1) # num_agents by slots*enc dim                
    return self.sec_pos_net(pos_enc)       

  
  def forward_mfp(self, hist:torch.Tensor, nbrs:torch.Tensor, masks:torch.Tensor, context:Any, 
                  nbrs_info:List, fut:torch.Tensor, bStepByStep:bool, 
                  use_forcing:Optional[Union[None,int]]=None) -> Tuple[List[torch.Tensor], Any]:
    """Forward propagation function for the MFP
    
    Computes dynamic state encoding with precomputed attention tensor and the 
    RNN based encoding.
    Args:
      hist: Trajectory history.
      nbrs: Neighbors.
      masks: Neighbors mask.
      context: contextual information in image form (if used).
      nbrs_info: information as to which other agents are neighbors.
      fut: Future Trajectory.
      bStepByStep: During rollout, interactive or independent.
      use_forcing: Teacher-forcing or classmate forcing.

    Returns:
      fut_pred: a list of predictions, one for each mode.
      modes_pred: prediction over latent modes.    
    """
    use_forcing = self.use_forcing if use_forcing==None else use_forcing

    # Normalize to reference position.
    ref_pos = hist[-1,:,:]
    hist = hist - ref_pos.view(1,-1,2)
    
    # Encode history trajectories.
    if isinstance(self.enc_lstm, torch.nn.modules.rnn.GRU):
      _, hist_enc = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
    else:
      _,(hist_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist))) #hist torch.Size([16, 128, 2])

    if self.use_gru:
      hist_enc = hist_enc.permute(1,0,2).contiguous()
      hist_enc = self.leaky_relu(self.dyn_emb( hist_enc.view(hist_enc.shape[0], -1) ))
    else:
      hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2]))) #torch.Size([128, 32])

    num_nbrs = sum([len(nbs) for nb_id, nbs in nbrs_info[0].items() ])      
    if num_nbrs > 0:
      nbrs_ref_pos = nbrs[-1,:,:]
      nbrs = nbrs - nbrs_ref_pos.view(1,-1,2) # normalize

      # Forward pass for all neighbors.
      if isinstance(self.enc_lstm, torch.nn.modules.rnn.GRU):
        _, nbrs_enc = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
      else:
        _, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))

      if self.use_gru:
        nbrs_enc = nbrs_enc.permute(1,0,2).contiguous()
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[0], -1)
      else:
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])
  
      attens = self.rbf_state_enc_get_attens(nbrs_enc, ref_pos, nbrs_info[0])            
      nbr_atten_enc = self.rbf_state_enc_hist_fwd(attens, nbrs_enc, nbrs_info[0])

    else: # if have no neighbors
      attens = None # type: ignore
      nbr_atten_enc = torch.zeros( 1, self.nbr_atten_embedding_size, dtype=torch.float32, device=masks.device )

    if self.use_context: #context encoding
      context_enc = self.relu(self.context_conv( context ))        
      context_enc = self.context_maxpool( self.context_conv2( context_enc ))
      context_enc = self.relu(self.context_conv3(context_enc))            
      context_enc = self.context_fc( context_enc.view( context_enc.shape[0], -1) )
      
      enc = torch.cat((nbr_atten_enc, hist_enc, context_enc),1)
    else:
      enc = torch.cat((nbr_atten_enc, hist_enc),1)
    # e.g. nbr_atten_enc: [num_agents by 80], hist_enc: [num_agents by 32], enc would be [num_agents by 112]
    
    ######################################################################################################      
    modes_pred = None if self.modes==1 else self.softmax(self.op_modes(enc))
    fut_pred = self.decode(enc, attens, nbrs_info[0], ref_pos, fut, bStepByStep, use_forcing)      
    return fut_pred, modes_pred
 
  def decode(self, enc: torch.Tensor, attens:List, nbrs_info_this:List, ref_pos:torch.Tensor, fut:torch.Tensor, bStepByStep:bool, use_forcing:Any ) -> List[torch.Tensor]:    
    """Decode the future trajectory using RNNs.
    
    Given computed feature vector, decode the future with multimodes, using
    dynamic attention and either interactive or non-interactive rollouts.
    Args:
      enc: encoded features, one per agent.
      attens: attentional weights, list of objs, each with dimenstion of [8 x 4] (e.g.)
      nbrs_info_this: information on who are the neighbors
      ref_pos: the current postion (reference position) of the agents.
      fut: future trajectory (only useful for teacher or classmate forcing)
      bStepByStep: interactive or non-interactive rollout
      use_forcing: 0: None. 1: Teacher-forcing. 2: classmate forcing.

    Returns:
      fut_pred: a list of predictions, one for each mode.
      modes_pred: prediction over latent modes.    
    """
    if not bStepByStep: # Non-interactive rollouts
      enc = enc.repeat(self.out_length, 1, 1)
      pos_enc = torch.zeros( self.out_length, enc.shape[1], self.posi_enc_dim+self.posi_enc_ego_dim, device=enc.device )
      enc2 = torch.cat( (enc, pos_enc), dim=2)                
      fut_preds = []
      for k in range(self.modes):
        h_dec, _ = self.dec_lstm[k](enc2)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op[k](h_dec)
        fut_pred = fut_pred.permute(1, 0, 2) #torch.Size([nSteps, num_agents, 5])

        fut_pred = Gaussian2d(fut_pred)
        fut_preds.append(fut_pred)            
      return fut_preds      
    else:
      batch_sz =  enc.shape[0]
      inds = []
      chunks = []
      for n in range(len(nbrs_info_this)):                  
        chunks.append( len(nbrs_info_this[n]) )
        for nbr in nbrs_info_this[n]:
          inds.append(nbr[0])
      flat_index = torch.LongTensor(inds).to(ref_pos.device) # type: ignore 
      
      fut_preds = []
      for k in range(self.modes):
        direc = 2 if self.bi_direc else 1
        hidden = torch.zeros(self.num_layers*direc, batch_sz, self.decoder_size).to(fut.device)
        preds: List[torch.Tensor] = []
        for t in range(self.out_length):
          if t == 0: # Intial timestep.
            if use_forcing == 0:                          
              pred_fut_t =  torch.zeros_like(fut[t,:,:])
              ego_fut_t = pred_fut_t
            elif use_forcing == 1:
              pred_fut_t = fut[t,:,:]
              ego_fut_t = pred_fut_t
            else:
              pred_fut_t = fut[t,:,:]
              ego_fut_t =  torch.zeros_like(pred_fut_t)
          else:
            if use_forcing == 0:
              pred_fut_t = preds[-1][:,:,:2].squeeze()
              ego_fut_t = pred_fut_t
            elif use_forcing == 1:
              pred_fut_t = fut[t,:,:]
              ego_fut_t = pred_fut_t
            else:
              pred_fut_t = fut[t,:,:]
              ego_fut_t = preds[-1][:,:,:2]

          if attens == None:
            pos_enc =  torch.zeros(batch_sz, self.posi_enc_dim, device=enc.device )
          else:
            pos_enc = self.rbf_state_enc_pos_fwd(attens, ref_pos, pred_fut_t, flat_index, chunks )
          
          enc_large = torch.cat( ( enc.view(1,enc.shape[0],enc.shape[1]), 
                                   pos_enc.view(1,batch_sz, self.posi_enc_dim),
                                   ego_fut_t.view(1, batch_sz, self.posi_enc_ego_dim ) ), dim=2 )

          out, hidden = self.dec_lstm[k]( enc_large, hidden)
          pred = Gaussian2d(self.op[k](out))
          preds.append( pred )
        fut_pred_k = torch.cat(preds,dim=0)
        fut_preds.append(fut_pred_k)
      return fut_preds
