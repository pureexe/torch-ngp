# nex360.py - an essential tools ported from NeX360 project

import torch
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import re

from nerf.provider import nerf_matrix_to_ngp


def rotate_camera(stock_c2w, angles, is_degree=True):
    """
    Rotate a camera around the origin
    @params stock_c2w - the c2w matrix of the current camera
    @params 
    @return c2w_mat - the rotated c2w matrix
    """
    r = Rotation.from_euler('xyz',angles,degrees=is_degree)
    rotation = r.as_matrix()
    w2c_mat = np.zeros((4,4),dtype = np.float32)
    w2c_mat[:3,:3] = rotation
    w2c_mat[3,3] = 1.0
    stock_w2c = np.linalg.inv(stock_c2w)
    translations = np.zeros_like(stock_w2c)
    translations[3, 3:4] = stock_w2c[3, 3:4]
    t_mat = stock_w2c - translations
    w2c_mat = t_mat @ w2c_mat
    w2c_mat = w2c_mat + translations
    c2w_mat = np.linalg.inv(w2c_mat)
    return c2w_mat

def get_uniform_sphere(num = 30, is_hemi = False, radius = 4.0, divider = 1.0):
    """
    @return: c2ws - camera position in Blender format
    """
    c2ws = []
    # @see https://stackoverflow.com/a/44164075
    if divider == 1.0:
        divider = 2 if is_hemi else 1
        indices = np.arange(0, num, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / (num * divider))
        theta = (np.pi * (1 + 5 **0.5) * indices)
        # create camera2world
        c2ws = []
        init_ext = np.eye(4,dtype=np.float32)
        init_ext[2,3] = radius #move z to 4
    for i in range(num):
      c2w_mat = rotate_camera(init_ext, [-phi[i], 0, 0], is_degree=False)
      c2w_mat = rotate_camera(c2w_mat, [0, 0, -theta[i]], is_degree=False) 
      c2w_mat = nerf_matrix_to_ngp(c2w_mat) #convert to NGP (aka. OpenGL) convention
      c2ws.append(c2w_mat[None])
    c2ws = np.concatenate(c2ws,axis=0)
    return c2ws

def stereographic(point3Ds):
  """
  project to the plane using spherical hamornic
  instant-ngp using OpenGL convention where y-up, then we use y as projector
  @params point3Ds - point in 3d space #[...,3]
  @return projected - projected point from stereographic projection #[...,2]
  """
  projected = np.zeros_like(point3Ds[...,:2])
  divider = point3Ds[...,1] + np.finfo(np.float32).eps #eps is a smallest number possible to avoid divide by zero problem
  projected[...,0] = point3Ds[...,0] / divider
  projected[...,1] = point3Ds[...,1] / divider
  return projected

def cos_between_vectors(u,v):
    # @params u,v should be shape [n,3] where  n is number of angle
    # @return cos theta from a \cdot b = |a|||b| cos theta
    
    # make u into unit vector
    u_norm = np.linalg.norm(u,axis = 1).reshape((-1,1))
    u_norm = np.broadcast_to(u_norm, (u_norm.shape[0],3))
    u_unit = u / u_norm
    # make v into unit vector
    v_norm = np.linalg.norm(v,axis = 1).reshape((-1,1))
    v_norm = np.broadcast_to(v_norm, (v_norm.shape[0],3))
    v_unit = v / v_norm
    
    # dot product to get the cos(theta)
    # from a \cdot b = |a||b| cos theta
    u_dot_v = np.matmul(u_unit.reshape(-1,1,3), v_unit.reshape(-1,3,1))
    u_dot_v = np.squeeze(u_dot_v)
    return u_dot_v

def position_probability(positions, target, sigma = 1):
    target_cast = np.broadcast_to(target, (positions.shape[0],3))
    distance = np.linalg.norm(positions - target_cast, axis=1)
    #negative distance softmax
    #@see https://stackoverflow.com/questions/23459707/how-to-convert-distance-into-probability
    prob = (np.exp(-(distance**2) / (sigma ** 2) ) ) / np.sum(np.exp(-(distance**2) / (sigma ** 2) ))
    # จูนว่า 0.9 กับ 10 ต้องเลือกอะไร+
    return prob

def position_probability_by_minimum(positions, target, min_angle=45, min_view=7, is_weight = False):
    target_cast = np.broadcast_to(target, (positions.shape[0],3))
    cos_angles = cos_between_vectors(positions, target_cast)
    threshold_cos = np.cos(min_angle * np.pi / 180.0)
    inds = np.flip(np.argsort(cos_angles))
    cos_angles = np.flip(np.sort(cos_angles))
    tail_id = min_view
    if tail_id > len(cos_angles):
        tail_id = len(cos_angles)
    while True:
        if cos_angles[tail_id - 1] <= threshold_cos or tail_id == len(cos_angles):
            break
        tail_id += 1
    prob = np.zeros_like(cos_angles)
    if is_weight:
        thres_prob = position_probability(positions[inds[:tail_id]], target, sigma = 1)
        prob[inds[:tail_id]] = thres_prob
    else:
        prob[inds[:tail_id]] = 1.0 / tail_id
    return inds[:tail_id], prob

def position_minimum_sortangle(positions, target, min_view=7, is_weight = False):
    target_cast = np.broadcast_to(target, (positions.shape[0],3))
    cos_angles = cos_between_vectors(positions, target_cast)
    inds = np.flip(np.argsort(cos_angles))
    tail_id = min_view    
    prob = np.zeros_like(cos_angles)
    if is_weight:
        thres_prob = position_probability(positions[inds[:tail_id]], target, sigma = 1)
        prob[:tail_id] = thres_prob
    else:
        prob[:tail_id] = 1.0 / tail_id
    return inds[:tail_id], prob[:tail_id]


def get_encoder_weights(location, num=30, radius=4.0, mode='closet'):
    """
    find the encoder weight to dataset class
    @params location - the center of input view camera in 3d space
    @return weights - shape [n,b] where n is the number of input view, closet: b=1, linear: b=2, delauney: b=3
    @return ids - shape [n,b] where n is the number of input view
    """
    input_pts = stereographic(location) #shape: (input_image,2)
    sphere_location = get_uniform_sphere(num, True, radius)[...,:3,3]
    sphere_pts = stereographic(sphere_location) #shape: (num,2)
    compare_shape = (input_pts.shape[0], num, 2) #shape: (input_image, num ,2)
    input_pts = np.broadcast_to(input_pts[...,None,:2], compare_shape)
    sphere_pts = np.broadcast_to(sphere_pts[None,:,:2], compare_shape)
    distance = np.linalg.norm((input_pts - sphere_pts),axis=-1)
    ids = np.argsort(distance)
    weights = np.sort(distance)
    if  mode.startswith('plane'):
      is_weight = 'weight' in mode
      reg_data = re.findall("\d+",mode)
      reg_data = [int(d) for d in reg_data]
      min_view = reg_data[0]
      if len(reg_data) > 1:
        raise Exception("Recieve too much parameter in plane mode")
      ids = []
      weights = []
      for i in range(location.shape[0]):        
        ret_id, ret_weight = position_minimum_sortangle(sphere_location, location[i], min_view=min_view, is_weight = True)
        ids.append(ret_id[None])
        weights.append(ret_weight[None])
      ids = np.concatenate(ids,axis=0)
      weights = np.concatenate(weights,axis=0)
    elif mode.startswith('angle'):
      is_weight = 'weight' in mode
      reg_data = re.findall("\d+",mode)
      reg_data = [int(d) for d in reg_data]
      min_angle = reg_data[0]
      min_view = 0
      if len(reg_data) == 2:
        min_view = reg_data[1]
      if len(reg_data) > 2:
        raise Exception("Recieve too much parameter in angle mode")
      raise NotImplementedError("NOT SURE HOW TO IMPLEMENT YET")
      location_length = location.shape[0]
      ids = []
      weights = []
      for i in range(location.shape[0]):
        id_ = np.zeros((100,),dtype=np.int32)
        weight_ = np.zeros((100,),dtype=np.float32)
        ret_id, ret_weight = position_probability_by_minimum(sphere_location, location[i], min_angle=min_angle, min_view=min_view, is_weight = True)
        id_ = id[:ret_id.shape[0]] = ret_id
        weight_ = id[:ret_weight.shape[0]] = ret_weight
        ids.append(id_[None])
      print("id_shape")
      print(id_.shape)
      print("weight_shape")
      print(weight_.shape)
      print("==========================")
      ids = ids[...,:2]
      weights = 1.0 - (weights[..., :2] / np.sum(weights[...,:2],axis=-1)[..., None])
      print(ids.shape)
      print(weights.shape)
      exit()
    elif mode == 'closet':
      ids = ids[...,:1]
      weights = np.ones_like(weights[...,:1])
    elif mode == 'linear':
      ids = ids[...,:2]
      weights = 1.0 - (weights[..., :2] / np.sum(weights[...,:2],axis=-1)[..., None])
    elif mode == 'delauney':
      raise NotImplementedError("Delauney mode doesn't implement yet")
    else:
      raise NotImplementedError('invalid encoder_weights mode')
    print("===============================")
    print(ids.shape)
    print(weights.shape)
    return ids, weights

def add_encoder_weights(dataset, num, mode='closet'):
  """
  set encoder_weights to NeRFDataset class
  """
  if mode == 1: mode = 'closet'
  if mode == 2: mode = 'linear'
  if mode == 3: mode = 'delauney'
  
  poses = dataset.poses[:,:3,3]
  is_preload = torch.is_tensor(poses)
  poses = poses.cpu().numpy() if is_preload else poses
  ids, weights = get_encoder_weights(poses, num, mode=mode)
  merge_weights = np.concatenate([ids[...,None], weights[...,None]], axis=-1)
  merge_weights = torch.from_numpy(merge_weights).cuda() if is_preload else merge_weights
  dataset.encoder_weights = merge_weights 
  return dataset