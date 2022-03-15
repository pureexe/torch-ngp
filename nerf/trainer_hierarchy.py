import torch
import os
import tensorboardX
import tqdm
import numpy as np


from .utils import Trainer as BaseTrainer
from .utils import get_rays

class Trainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        print("TRAINER: Hierachy mode trainer")
        super().__init__(*args, **kwargs)
        self.learning_rate = {
            'start': 5e-4,
            'final': 5e-5
        }
        self.max_epochs = 10000

    def train_step(self, data):
        images = data["image"] # [B, H, W, 3/4]
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]

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

        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, **self.conf)
        
        pred_rgb = outputs['rgb'][...,3:6]
        lvl0_rgb = outputs['rgb'][...,:3]

        loss = self.criterion(pred_rgb, gt_rgb) + self.criterion(lvl0_rgb, gt_rgb)

        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):
        images = data["image"] # [B, H, W, 3/4]
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]

        # sample rays 
        B, H, W, C = images.shape
        rays_o, rays_d, _ = get_rays(poses, intrinsics, H, W, -1)

        bg_color = torch.ones(3, device=images.device) # [3]
        # eval with fixed background color
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
            
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False,  **self.conf)
        
        pred_rgb = outputs['rgb'].reshape(B, H, W, -1) #[...,3:6]
        pred_depth = outputs['depth'].reshape(B, H, W)
        
        mse = torch.mean((pred_rgb - gt_rgb) ** 2)
        psnr = 10 * torch.log10(1.0 / mse)
        #loss = self.criterion(pred_rgb, gt_rgb)

        return pred_rgb, pred_depth, gt_rgb, psnr

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):  
        poses = data["pose"] # [B, 4, 4]
        intrinsics = data["intrinsic"] # [B, 3, 3]

        H, W = int(data['H'][0]), int(data['W'][0]) # get the target size...

        B = poses.shape[0]

        rays_o, rays_d, _ = get_rays(poses, intrinsics, H, W, -1)

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **self.conf)

        pred_rgb = outputs['rgb'].reshape(B, H, W, -1)# [...,3:6]
        pred_depth = outputs['depth'].reshape(B, H, W)

        return pred_rgb, pred_depth

    def save_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] >= self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def set_learning_rate(self):
        ''' set learning rate for network '''
        t = np.clip(self.epoch / self.max_epochs, 0, 1)
        if self.learning_rate['start'] == self.learning_rate['final']:
            rate = self.learning_rate['final']
        else:
            rate = np.exp(np.log(self.learning_rate['start']) * (1 - t) + np.log(self.learning_rate['final']) * t)
        # set learning rate off both optimizer
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = rate

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # update grid
        if self.model.cuda_ray:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state(self.conf['bound'])

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            self.local_step += 1
            self.global_step += 1
            
            data = self.prepare_data(data)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                for metric in self.metrics:
                    metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            for metric in self.metrics:
                self.log(metric.report(), style="red")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="train")
                metric.clear()

        if not self.scheduler_update_every_step:
           self.set_learning_rate()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def train(self, train_loader, valid_loader, max_epochs):
        self.max_epochs = max_epochs
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()