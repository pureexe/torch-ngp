import torch

from .utils import Trainer as BaseTrainer
from .utils import get_rays

class Trainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        print("TRAINER: Fibonacci trainer")
        super().__init__(*args, **kwargs)

    def train_step(self, data):
        images = data["image"] # [B, H, W, 3/4]
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]
        encoder_weights = data["encoder_weights"] #[B, I, 2]: I\in[1,3]

        # sample rays 
        B, H, W, C = images.shape
        rays_o, rays_d, inds = get_rays(poses, intrinsics, H, W, self.conf['num_rays'])
        images = torch.gather(images.reshape(B, -1, C), 1, torch.stack(C*[inds], -1)) # [B, N, 3/4]

        # train with random background color if using alpha mixing
        #bg_color = torch.ones(3, device=images.device) # [3], fixed white background
        bg_color = torch.rand(3, device=images.device) # [3], frame-wise random.
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, encoder_weights=encoder_weights, **self.conf)
    
        pred_rgb = outputs['rgb']

        loss = self.criterion(pred_rgb, gt_rgb)

        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):
        images = data["image"] # [B, H, W, 3/4]
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]
        encoder_weights = data["encoder_weights"] #[B, I, 2]: I\in[1,3]

        # sample rays 
        B, H, W, C = images.shape
        rays_o, rays_d, _ = get_rays(poses, intrinsics, H, W, -1)

        bg_color = torch.ones(3, device=images.device) # [3]
        # eval with fixed background color
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
            
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, encoder_weights=encoder_weights, **self.conf)

        pred_rgb = outputs['rgb'].reshape(B, H, W, -1)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb)

        return pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):  
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]
        encoder_weights = data["encoder_weights"] #[B, I, 2]: I\in[1,3]

        H, W = int(data['H'][0]), int(data['W'][0]) # get the target size...

        B = poses.shape[0]

        rays_o, rays_d, _ = get_rays(poses, intrinsics, H, W, -1)

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, encoder_weights=encoder_weights, **self.conf)

        pred_rgb = outputs['rgb'].reshape(B, H, W, -1)
        pred_depth = outputs['depth'].reshape(B, H, W)

        return pred_rgb, pred_depth