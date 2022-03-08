# nex360.py - an essential tools ported from NeX360 project

import torch
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

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


def get_encoder_weights(location, num=30, radius=4.0, mode='closet'):
    """
    find the encoder weight to dataset class
    @params location - the center of input view camera in 3d space
    @return weights - shape [n,b] where n is the number of input view, closet: b=1, linear: b=2, delauney: b=3
    @return ids - shape [n,b] where n is the number of input view
    """
    input_pts = stereographic(location) #shape: (input_image,2)
    sphere_pts = stereographic(get_uniform_sphere(num, True, radius)[...,:3,3]) #shape: (num,2)
    compare_shape = (input_pts.shape[0], num, 2) #shape: (input_image, num ,2)
    input_pts = np.broadcast_to(input_pts[...,None,:2], compare_shape)
    sphere_pts = np.broadcast_to(sphere_pts[None,:,:2], compare_shape)
    distance = np.linalg.norm((input_pts - sphere_pts),axis=-1)
    ids = np.argsort(distance)
    weights = np.sort(distance)
    if mode == 'closet':
      ids = ids[...,:1]
      weights = np.ones_like(weights[...,:1])
    elif mode == 'linear':
      ids = ids[...,:2]
      weights = 1.0 - (weights[..., :2] / np.sum(weights[...,:2],axis=-1)[..., None])
    elif mode == 'delauney':
      raise NotImplementedError("Delauney mode doesn't implement yet")
    else:
      raise NotImplementedError('invalid encoder_weights mode')

    return ids, weights

def add_encoder_weights(dataset, mode='closet'):
  """
  set encoder_weights to NeRFDataset class
  """
  if mode == 1: mode = 'closet'
  if mode == 2: mode = 'linear'
  if mode == 3: mode = 'delauney'
  
  poses = dataset.poses[:,:3,3]
  is_preload = torch.is_tensor(poses)
  poses = poses.cpu().numpy() if is_preload else poses
  ids, weights = get_encoder_weights(poses)
  merge_weights = np.concatenate([ids[...,None], weights[...,None]], axis=-1)
  merge_weights = torch.from_numpy(merge_weights).cuda() if is_preload else merge_weights
  dataset.encoder_weights = merge_weights 
  return dataset