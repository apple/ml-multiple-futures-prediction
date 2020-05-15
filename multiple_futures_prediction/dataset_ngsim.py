#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019-2020 Apple Inc. All Rights Reserved.
#

from typing import List, Set, Dict, Tuple, Optional, Union, Any
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
import pickle
import os
import cv2

# Dataset for pytorch training
class NgsimDataset(Dataset):
  def __init__(self, mat_file:str, t_h:int=30, t_f:int=50, d_s:int = 2, 
   	                enc_size:int=64, use_gru:bool=False, self_norm:bool=False, 
                    data_aug:bool=False, use_context:bool=False, nbr_search_depth:int= 3, 
                    ds_seed:int=1234) -> None:
    self.D = scp.loadmat(mat_file)['traj']
    self.T = scp.loadmat(mat_file)['tracks']
    self.t_h = t_h  # length of track history
    self.t_f = t_f  # length of predicted trajectory
    self.d_s = d_s  # down sampling rate of all sequences
    self.enc_size = enc_size # size of encoder LSTM
    self.grid_size = (13,3) # size of context grid
    self.enc_fac = 2 if use_gru else 1
    self.self_norm = self_norm
    self.data_aug = data_aug
    self.noise = np.array([[0.5, 2.0]])
    self.dt = 0.1*self.d_s
    self.ft_to_m = 0.3048
    self.use_context = use_context
    if self.use_context:
      self.maps = pickle.load(open('data/maps.pkl', 'rb'))
    self.nbr_search_depth = nbr_search_depth

    cache_file = 'multiple_futures_prediction/ngsim_data/NgsimIndex_%s.p'%os.path.basename(mat_file) 
    #build index of [dataset (0 based), veh_id_0b, frame(time)] into a dictionary
    if not os.path.exists(cache_file):          
      self.Index = {}
      print('building index...')
      for i, row in enumerate(self.D):
        key = (int(row[0]-1), int(row[1]-1), int(row[2]))
        self.Index[key] = i
      print('build index done')
      pickle.dump( self.Index, open(cache_file,'wb'))
    else:
      self.Index = pickle.load( open(cache_file,'rb'))

    self.ind_random = np.arange(len(self.D))
    self.seed = ds_seed
    np.random.seed(self.seed)
    np.random.shuffle(self.ind_random)

  def __len__(self) -> int:
    return len(self.D)

  def convert_pt(self, pt: np.ndarray, dsId0b: int) -> np.ndarray:
    """Convert a point from abs coords to pixel coords.
    Args:
      pt is a 2d x,y coordinate
      dsId0b - data set id 0 index based
    """
    ft_per_pixel = self.maps[dsId0b]['ft_per_pixel']
    return np.array([int((np.round(pt[0])-self.maps[dsId0b]['x0'])/ft_per_pixel), 
                    int((np.round(pt[1]) - self.maps[dsId0b]['y0'])/ft_per_pixel)])

  def compute_vel_theta(self, hist: torch.Tensor, frac: Optional[float]=0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate velocity and orientation from history trajectory."""
    if hist.shape[0] <= 1:
      return np.array([0.0]), np.array([0.0])
    else:   
      total_wts = 0.0
      counter = 0.0
      vel = theta = 0.0        
      for t in range(hist.shape[0]-1,0,-1):
        counter += 1.0
        wt = np.power(frac, counter)
        total_wts += wt            
        diff = hist[t,:] - hist[t-1,:]            
        vel     += wt*np.linalg.norm(diff)*self.ft_to_m/self.dt
        theta   += wt*np.arctan2(diff[1],diff[0])
      return np.array([vel/total_wts]),  np.array([theta/total_wts])    
  
  def find_neighbors(self, dsID_0b: int, vehId_0b: int, frame: int) -> Dict:
    """Find a list of neighbors w.r.t. self."""
    key = ( int(dsID_0b), int(vehId_0b), int(frame))
    if key not in self.Index:
      return {}
    
    idx = self.Index[key]
    grid1b = self.D[idx,8:]
    nonzero = np.nonzero(grid1b)[0]
    return {vehId_0b: list(zip( grid1b[nonzero].astype(np.int64)-1, nonzero)) }

  def __getitem__(self, idx_: int) -> Tuple[ List, List, Dict, Union[None, np.ndarray] ] :        
    idx = self.ind_random[idx_]
    dsId_1b = self.D[idx, 0].astype(int)
    vehId_1b = self.D[idx, 1].astype(int)
    dsId_0b = dsId_1b-1
    vehId_0b = vehId_1b-1
    t = self.D[idx, 2]
    grid = self.D[idx,8:] #1-based

    ids = {} #0-based keys
    leafs = [vehId_0b]
    for _ in range( self.nbr_search_depth ):
      new_leafs = []
      for id0b in leafs:
        nbr_dict = self.find_neighbors(dsId_0b, id0b, t)                
        if len(nbr_dict) > 0:
          ids.update( nbr_dict )
          if len(nbr_dict[id0b]) > 0:
            nbr_id0b = list( zip(*nbr_dict[id0b]))[0]
            new_leafs.extend (  nbr_id0b )
      leafs = np.unique( new_leafs )
      leafs = [x for x in leafs if x not in ids]      
    
    sorted_keys = sorted(ids.keys())  # e.g. [1, 3, 4, 5, ... , 74]   
    id_map = {key: value for (key, value) in zip(sorted_keys, np.arange(len(sorted_keys)).tolist()) } 
    #obj id to index within a batch
    sz = len(ids)
    assert sz > 0

    hist = []
    fut = []
    neighbors: Dict[int,List] = {} # key is batch ind, followed by a list of (batch_ind, ego_id, grid/nbr ind)
    
    for ind, vehId0b in enumerate(sorted_keys):
      hist.append( self.getHistory0b(vehId0b,t,dsId_0b) )    # no normalization         
      fut.append(  self.getFuture(vehId0b+1,t,dsId_0b+1 ) )  #subtract off ref pos
      neighbors[ind] = []
      for v_id, nbr_ind in ids[ vehId0b ]:
        if v_id not in id_map:
          k2 = -1
        else:
          k2 = id_map[v_id]
          neighbors[ind].append( (k2, v_id, nbr_ind) )
    
    if self.use_context:
      x_range_ft = np.array([-15, 15])
      y_range_ft = np.array([-30, 300])

      pad = int(np.ceil(300/self.maps[dsId_1b-1]['ft_per_pixel'])) # max of all ranges
      
      if not 'im_color' in self.maps[dsId_1b-1]:
        im_big = np.pad(self.maps[dsId_1b-1]['im'], ((pad, pad), (pad, pad)), 'constant', constant_values= 0.0)
        self.maps[dsId_1b-1]['im_color'] = (im_big[np.newaxis,...].repeat(3, axis=0)*255.0).astype(np.uint8)
      
      im = self.maps[dsId_1b-1]['im_color']
      height, width = im.shape[1:]      
      
      ref_pos = self.D[idx,3:5]
      im_x, im_y = self.convert_pt( ref_pos, dsId_1b-1 )
      im_x += pad
      im_y += pad

      x_range = (x_range_ft/self.maps[dsId_1b-1]['ft_per_pixel']).astype(int)
      y_range = (y_range_ft/self.maps[dsId_1b-1]['ft_per_pixel']).astype(int)
      
      x_range[0] = np.maximum( 0,         x_range[0]+im_x )
      x_range[1] = np.minimum( width-1,   x_range[1]+im_x )
      y_range[0] = np.maximum( 0,         y_range[0]+im_y )
      y_range[1] = np.minimum( height-1,  y_range[1]+im_y )

      im_crop = np.ascontiguousarray(im[:, y_range[0]:y_range[1], x_range[0]:x_range[1]].transpose((1,2,0)))
      im_crop[:,:,[0, 1]] = 0

      for _, other in neighbors.items():
        if len(other) == 0:
          continue
        for k in range( len(other)-1 ):
          x1, y1 = self.convert_pt( other[k]+ref_pos, dsId_1b-1 )
          x2, y2 = self.convert_pt( other[k+1]+ref_pos, dsId_1b-1 )
          x1+=pad; y1+=pad; x2+=pad; y2+=pad
          cv2.line(im_crop,(x1-x_range[0],y1-y_range[0]),(x2-x_range[0],y2-y_range[0]), (255, 0, 0), 2 )
        
        x, y = self.convert_pt( other[-1]+ref_pos, dsId_1b-1 )
        x+=pad; y+=pad
        cv2.circle(im_crop, (x-x_range[0], y-y_range[0]), 4, (255, 0, 0), -1)

      for k in range( len(hist)-1 ):
        x1, y1 = self.convert_pt( hist[k]+ref_pos, dsId_1b-1 )
        x2, y2 = self.convert_pt( hist[k+1]+ref_pos, dsId_1b-1 )
        x1+=pad; y1+=pad; x2+=pad; y2+=pad
        cv2.line(im_crop,(x1-x_range[0],y1-y_range[0]),(x2-x_range[0],y2-y_range[0]), (0, 255, 0), 3 )

      cv2.circle(im_crop, (im_x-x_range[0], im_y-y_range[0]), 5, (0, 255, 0), -1)
      assert im_crop.shape == (660,60,3)
      im_crop = im_crop.transpose((2,0,1))
    else:
      im_crop = None    
    return hist, fut, neighbors, im_crop  # neighbors is a list of all vehicles in the batch
  
  def getHistory(self, vehId: int, t: int, refVehId: int, dsId: int) -> np.ndarray:
    """Get trajectory history. VehId and refVehId are 1-based."""
    if vehId == 0:
      return np.empty([0,2])
    else:
      if self.T.shape[1]<=vehId-1:
        return np.empty([0,2])
      vehTrack = self.T[dsId-1][vehId-1].transpose()            
      if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
         return np.empty([0,2])
      else:
        refTrack = self.T[dsId-1][refVehId-1].transpose()
        found = np.where(refTrack[:,0]==t)
        refPos = refTrack[found][0,1:3]

        stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
        enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
        hist = vehTrack[stpt:enpt:self.d_s,1:3]-refPos

        if self.data_aug:
          hist += np.random.randn( hist.shape[0],hist.shape[1] )*self.noise

      if len(hist) < self.t_h//self.d_s + 1:
        return np.empty([0,2])
      return hist

  def getFuture(self, vehId:int, t:int, dsId: int) -> np.ndarray :
    """Get future trajectory. VehId and dsId are 1-based."""
    vehTrack = self.T[dsId-1][vehId-1].transpose()
    refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
    stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
    enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
    fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
    return fut

  def getHistory0b(self, vehId:int, t:int, dsId:int ) -> np.ndarray :
    """Get track history trajectory. VehId and dsId are zero-based.
    No normalizations are performed.
    """
    if vehId < 0:
      return np.empty([0,2])
    else:
      if vehId >= self.T.shape[1]:
        return np.empty([0,2])                
      vehTrack = self.T[dsId][vehId].transpose()
      if vehTrack.size==0 or np.argwhere(vehTrack[:, 0] == t).size==0:
         return np.empty([0,2])
      else:
        stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
        enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
        hist = vehTrack[stpt:enpt:self.d_s,1:3]            
      if len(hist) < self.t_h//self.d_s + 1:
        return np.empty([0,2])
      return hist

  def getFuture0b(self, vehId:int, t:int, dsId:int, refPos:int) -> np.ndarray :
    """Get track future trajectory. VehId and dsId are zero-based."""
    vehTrack = self.T[dsId][vehId].transpose()        
    stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
    enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
    fut = vehTrack[stpt:enpt:self.d_s,1:3]-refPos
    return fut

  def collate_fn(self, samples: List[Any]) -> Tuple[Any,Any,Any,Any,Any,Union[Any,None],Any] :
    """Prepare a batch suitable for MFP training."""
    nbr_batch_size = 0
    num_samples = 0
    for _,_,nbrs,im_crop in samples:
      nbr_batch_size +=  sum([len(nbr) for nbr in nbrs.values() ])      
      num_samples += len(nbrs)

    maxlen = self.t_h//self.d_s + 1
    if nbr_batch_size <= 0:      
      nbrs_batch = torch.zeros(maxlen,1,2)
    else:
      nbrs_batch = torch.zeros(maxlen,nbr_batch_size,2)
    
    pos = [0, 0]
    nbr_inds_batch = torch.zeros( num_samples, self.grid_size[1],self.grid_size[0], self.enc_size*self.enc_fac)
    nbr_inds_batch = nbr_inds_batch.byte()

    hist_batch = torch.zeros(maxlen, num_samples, 2)  #e.g. (31, 41, 2)
    fut_batch       = torch.zeros(self.t_f//self.d_s, num_samples, 2)
    mask_batch   = torch.zeros(self.t_f//self.d_s, num_samples, 2)    
    if self.use_context:
      context_batch = torch.zeros(num_samples, im_crop.shape[0], im_crop.shape[1], im_crop.shape[2] )
    else:
      context_batch: Union[None, torch.Tensor] = None # type: ignore

    nbrs_infos = []
    count = 0
    samples_so_far = 0
    for sampleId,(hist, fut, nbrs, context) in enumerate(samples):            
      num = len(nbrs)      
      for j in range(num):
        hist_batch[0:len(hist[j]), samples_so_far+j, :] = torch.from_numpy(hist[j])
        fut_batch[0:len(fut[j]), samples_so_far+j, :] = torch.from_numpy(fut[j])
        mask_batch[0:len(fut[j]),samples_so_far+j,:] = 1                
      samples_so_far += num

      nbrs_infos.append(nbrs)

      if self.use_context:
        context_batch[sampleId,:,:,:] = torch.from_numpy(context)                

      # nbrs is a dictionary of key to list of nbr (batch_index, veh_id, grid_ind)
      for batch_ind, list_of_nbr in nbrs.items():
        for batch_id, vehid, grid_ind in list_of_nbr:          
          if batch_id >= 0:
            nbr_hist = hist[batch_id]                                    
            nbrs_batch[0:len(nbr_hist),count,:] = torch.from_numpy( nbr_hist )
            pos[0] = grid_ind % self.grid_size[0]
            pos[1] = grid_ind // self.grid_size[0]
            nbr_inds_batch[batch_ind,pos[1],pos[0],:] = torch.ones(self.enc_size*self.enc_fac).byte()
            count+=1

    return (hist_batch, nbrs_batch, nbr_inds_batch, fut_batch, mask_batch, context_batch, nbrs_infos)
