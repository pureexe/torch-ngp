import torch

from nerf.provider import NeRFDataset
from nerf.utils import *

import argparse,time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--upsample_steps', type=int, default=128)
    parser.add_argument('--max_ray_batch', type=int, default=4096) # lower if OOM
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--plane3', action='store_true', help="use 3 plane projection")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")
    
    parser.add_argument('--mode', type=str, default='colmap', help="dataset mode, supports (colmap, blender)")
    # the default setting for fox.
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box(-bound, bound)")

    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")

    opt = parser.parse_args()

    print(opt)

    if opt.plane3:
        from nerf.network_plane3 import NeRFNetwork
        print('NeRF: plane3')
    elif opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from nerf.network_ff import NeRFNetwork
        print('NeRF: Fully-Fused')
    elif opt.tcnn:
        from nerf.network_tcnn import NeRFNetwork
        print('NeRF: TCNN')
    else:
        from nerf.network import NeRFNetwork
        print('NeRF: torch implemented')
        
    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding="hashgrid", encoding_dir="sphere_harmonics", 
        num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64, 
        cuda_ray=opt.cuda_ray,
    )

    print(model)
    training_start = time.time()
    trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='latest',)
    
    # save mesh
    #trainer.save_mesh()
    test_dataset = NeRFDataset(opt.path, type='test', mode=opt.mode, scale=opt.scale)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    trainer.test(test_loader)
    print(">>>>> finished testing in {:6f} seconds <<<<<<".format(time.time() - training_start))
