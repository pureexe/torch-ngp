import torch

from nerf.provider import NeRFDataset
from nerf.utils import *

import argparse, os

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=310)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--upsample_steps', type=int, default=128)
    parser.add_argument('--max_ray_batch', type=int, default=4096)
    parser.add_argument('--hash_size', type=int, default=19, help="hashmap size in term of 2^n")
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")
    parser.add_argument('--plane3', action='store_true', help="use 3 plane projection")

    parser.add_argument('--mode', type=str, default='colmap', help="dataset mode, supports (colmap, blender)")
    # the default setting for fox.
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box(-bound, bound)")

    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--skip_if_exist', action='store_true', help="if thje work space")

    opt = parser.parse_args()

    print(opt)
    if opt.skip_if_exist and os.path.exists(opt.workspace):
        print("Skiping existed experiment: ", opt.workspace)
        exit()
    else:
        print(opt.workspace)
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

    train_dataset = NeRFDataset(opt.path, type='train', mode=opt.mode, scale=opt.scale)
    valid_dataset = NeRFDataset(opt.path, type='val', mode=opt.mode,  scale=opt.scale)
    #valid_dataset = NeRFDataset(opt.path, type='val', mode=opt.mode, downscale=2, scale=opt.scale) #PURE: previously down scale on evaluation by 2

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)
    
    model = NeRFNetwork(
        encoding="hashgrid", encoding_dir="sphere_harmonics", 
        num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64, 
        cuda_ray=opt.cuda_ray,
        log2_hashmap_size=opt.hash_size
    )

    #model = NeRFNetwork(encoding="frequency", encoding_dir="frequency", num_layers=4, hidden_dim=256, geo_feat_dim=256, num_layers_color=4, hidden_dim_color=128)

    print(model)

    #criterion = torch.nn.SmoothL1Loss()
    criterion = torch.nn.HuberLoss()
    #criterion = torch.nn.MSELoss()    
    print("INSTANT-NPG paper use HuberLoss i guess?")

    if hasattr(model, 'encoders'):
        encoding_params = []
        for k in model.encoders:
            encoding_params += list(k.parameters())
    else:
        encoding_params = list(model.encoder.parameters())
    optimizer = lambda model: torch.optim.Adam([
        {'name': 'encoding', 'params': encoding_params},
        {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
    ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    #scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.33)
    mile_stone = opt.num_epoch // 4
    scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[mile_stone, mile_stone * 2, mile_stone * 3], gamma=0.33)
    #scheduler = lambda optimizer: optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.962709)


    print(model)
    trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=opt.eval_interval)
    training_start = time.time()
    trainer.train(train_loader, valid_loader, opt.num_epoch)
    print(">>>>> finished training in {:6f} seconds <<<<<<".format(time.time() - training_start))
    # test dataset
    #trainer.save_mesh()
    test_dataset = NeRFDataset(opt.path, type='test', mode=opt.mode, scale=opt.scale)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    trainer.test(test_loader)
