import torch
import torch.optim as optim
import argparse

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import seed_everything, PSNRMeter
from nerf.nex360 import add_encoder_weights

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    ### training options
    parser.add_argument('--num_epochs', type=int, default=310) #pure
    parser.add_argument('--eval_interval',type=int, default=50) #pure
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    # (only valid when not using --cuda_ray)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--upsample_steps', type=int, default=128)
    parser.add_argument('--max_ray_batch', type=int, default=4096)
    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")
    parser.add_argument('--plane3', action='store_true', help="use plane3 backend") #pure
    parser.add_argument('--fibonacci', type=int, default=0, help="use fibonacci lattice plane if more than 0") #pure
    parser.add_argument('--blend_lattice', type=int, default=1, help="blend n [1,2,3] lattice (default: 1). note that 3 will be delauney version") #pure
    parser.add_argument('--global_tri', action='store_true', help="if enable, it will append tri plane project to encoder") #pure
    parser.add_argument('--refiner_ratio',type=float, default=-1, help="use fiborefiner network if >= 0") #pure
    parser.add_argument('--train_plane_mode',type=str, default="", help="train plane picking scheme") #pure
    parser.add_argument('--og', action='store_true', help="use original nerf")
    parser.add_argument('--learning_rate', type=float, default=1e-2)

    ### dataset options
    parser.add_argument('--mode', type=str, default='colmap', help="dataset mode, supports (colmap, blender)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, fasten training but use more GPU memory")
    # (default is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box(-bound, bound)")
    #multiplane option
    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=90, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    opt = parser.parse_args()
    print(opt)
    
    seed_everything(opt.seed)
    
    if opt.og:
        from nerf.network_nerf import NeRFNetwork
        from nerf.utils import Trainer
    elif opt.refiner_ratio >= 0:
        print('Network: FiboRefiner')
        from nerf.network_fiborefiner import NeRFNetwork
        from nerf.trainer_fibonacci import Trainer
    elif opt.fibonacci > 0:
        print('Network: Fibonacci')
        from nerf.network_fibonacci import NeRFNetwork
        from nerf.trainer_fibonacci import Trainer
    elif opt.plane3:
        print('Network: Plane3')
        from nerf.network_plane3 import NeRFNetwork
        from nerf.utils import Trainer
    elif opt.ff:
        print('Network: FullyFused')
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from nerf.network_ff import NeRFNetwork
        from nerf.utils import Trainer
    elif opt.tcnn:
        print('Network: TCNN')
        from nerf.network_tcnn import NeRFNetwork
        from nerf.utils import Trainer
    else:
        print('Network: NeRF')
        from nerf.network import NeRFNetwork
        from nerf.utils import Trainer

    model = NeRFNetwork(
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        fibonacci=opt.fibonacci,
        blend_lattice=opt.blend_lattice,
        global_tri=opt.global_tri,
        refiner_ratio=opt.refiner_ratio
    )
    
    print(model)

    criterion = torch.nn.HuberLoss(delta=0.1)

    ### test mode
    if opt.test:

        trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=[PSNRMeter()], use_checkpoint='latest')

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_dataset = NeRFDataset(opt.path, type='test', mode=opt.mode, scale=opt.scale, preload=opt.preload)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

            if opt.mode == 'blender':
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            else:
                trainer.test(test_loader) # colmap doesn't have gt, so just test.
    
    else:

        optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': list(model.encoder.parameters())},
            {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
        ], lr=opt.learning_rate, betas=(0.9, 0.99), eps=1e-15)

        # need different milestones for GUI/CMD mode.
        quatuer_stone = opt.num_epochs // 4#pure
        scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 1500, 2000] if opt.gui else [quatuer_stone, quatuer_stone*2, quatuer_stone*3], gamma=0.33)

        trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, metrics=[PSNRMeter()], use_checkpoint='latest', eval_interval=10)

        # need different dataset type for GUI/CMD mode.

        if opt.gui:
            train_dataset = NeRFDataset(opt.path, type='all', mode=opt.mode, scale=opt.scale, preload=opt.preload)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
            trainer.train_loader = train_loader # attach dataloader to trainer

            if opt.fibonacci > 0:
                raise NotImplementedError("Currently, fibonacci didn't support GUI training")

            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            train_dataset = NeRFDataset(opt.path, type='train', mode=opt.mode, scale=opt.scale, preload=opt.preload)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
            valid_dataset = NeRFDataset(opt.path, type='val', mode=opt.mode, downscale=2, scale=opt.scale, preload=opt.preload)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

            trainer.train(train_loader, valid_loader, opt.num_epochs)

            # also test
            test_dataset = NeRFDataset(opt.path, type='test', mode=opt.mode, scale=opt.scale, preload=opt.preload)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
            
            if opt.mode == 'blender':
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            else:
                trainer.test(test_loader) # colmap doesn't have gt, so just test.