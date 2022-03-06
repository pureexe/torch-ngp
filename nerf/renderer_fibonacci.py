import torch

from .renderer import NeRFRenderer as BaseRenderer
from .renderer import near_far_from_bound, sample_pdf

class NeRFRenderer(BaseRenderer):
        def __init__(self, *args, **kwargs):
            print("Renderer: fibonacci renderer")
            super().__init__(*args,**kwargs)

        def run(self, rays_o, rays_d, bound, num_steps, upsample_steps, bg_color, perturb, encoder_weights):
            """
            render using nerf sampling
            @params encoder_weights - shape:[B,I,2]
            @return rendered - see run or run_hierarchy_plane for more info, usually, contain rgb_image and depth
            """
            rendererd = None
            if self.training:
                # we render only a ray from one plane on training step
                idx = encoder_weights[:,:,1].multinomial(1,replacement=False)[...,0] #pick 1 plane from encoder_weight shape:[B]
                plane_id = encoder_weights[:,:,0][idx][0] # shape:[B]
                rendered = self.run_hierarchy_plane(rays_o, rays_d, bound, num_steps, upsample_steps, bg_color, perturb, plane_id)
            else: 
                # similar to NeX360, we blend the result from multiple plane
                for i in range(encoder_weights.shape[-2]):
                    plane_id = encoder_weights[:,i,0]
                    plane_weight = encoder_weights[:,i,1]
                    rendered_plane = self.run_hierarchy_plane(rays_o, rays_d, bound, num_steps, upsample_steps, bg_color, perturb, plane_id)
                    rendered_plane = list(rendered_plane)
                    # weight the color
                    for j in range(len(rendered_plane)):
                        if(rendered_plane[j].shape == 3):
                            rendered_plane[j] = rendered_plane[j] * plane_weight[...,None].expand(-1,-1,rendered[j].shape[2])
                        else:
                            rendered_plane[j] = rendered_plane[j] * plane_weight
                    # add the color back to the rendered component
                    if i == 0:
                        rendered = rendered_plane
                    else:
                        for j in range(len(rendered_plane)):
                            rendered[j] = rendered[j] + rendered_plane[j]
            return rendered                    


        def run_hierarchy_plane(self, rays_o, rays_d, bound, num_steps, upsample_steps, bg_color, perturb, plane_id):
            # rays_o, rays_d: [B, N, 3], assumes B == 1
            # bg_color: [3] in range [0, 1]
            # plane_id: [B] shape 
            # return: image: [B, N, 3], depth: [B, N]

            B, N = rays_o.shape[:2]
            device = rays_o.device

            # sample steps
            near, far = near_far_from_bound(rays_o, rays_d, bound, type='cube')

            #print(f'near = {near.min().item()} ~ {near.max().item()}, far = {far.min().item()} ~ {far.max().item()}')

            z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0).unsqueeze(0) # [1, 1, T]
            z_vals = z_vals.expand((B, N, num_steps)) # [B, N, T]
            z_vals = near + (far - near) * z_vals # [B, N, T], in [near, far]

            # perturb z_vals
            sample_dist = (far - near) / num_steps
            if perturb:
                z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
                #z_vals = z_vals.clamp(near, far) # avoid out of bounds pts.

            # generate pts
            pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [B, N, 1, 3] * [B, N, T, 3] -> [B, N, T, 3]
            pts = pts.clamp(-bound, bound) # must be strictly inside the bounds, else lead to nan in hashgrid encoder!

            #print(f'pts {pts.shape} {pts.min().item()} ~ {pts.max().item()}')

            #plot_pointcloud(pts.reshape(-1, 3).detach().cpu().numpy())

            # query SDF and RGB
            dirs = rays_d.unsqueeze(-2).expand_as(pts)

            sigmas, rgbs = self(pts.reshape(B, -1, 3), dirs.reshape(B, -1, 3), bound=bound, plane_id=plane_id)

            rgbs = rgbs.reshape(B, N, num_steps, 3) # [B, N, T, 3]
            sigmas = sigmas.reshape(B, N, num_steps) # [B, N, T]

            # upsample z_vals (nerf-like)
            if upsample_steps > 0:
                with torch.no_grad():

                    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1] # [B, N, T-1]
                    deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[:, :, :1])], dim=-1)

                    alphas = 1 - torch.exp(-deltas * sigmas) # [B, N, T]
                    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-15], dim=-1) # [B, N, T+1]
                    weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[:, :, :-1] # [B, N, T]

                    # sample new z_vals
                    z_vals_mid = (z_vals[:, :, :-1] + 0.5 * deltas[:, :, :-1]) # [B, N, T-1]
                    new_z_vals = sample_pdf(z_vals_mid.reshape(B*N, -1), weights.reshape(B*N, -1)[:, 1:-1], upsample_steps, det=not self.training).detach() # [BN, t]
                    new_z_vals = new_z_vals.reshape(B, N, upsample_steps)

                    new_pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [B, N, 1, 3] * [B, N, t, 3] -> [B, N, t, 3]
                    new_pts = new_pts.clamp(-bound, bound)

                # only forward new points to save computation
                new_dirs = rays_d.unsqueeze(-2).expand_as(new_pts)
                new_sigmas, new_rgbs = self(new_pts.reshape(B, -1, 3), new_dirs.reshape(B, -1, 3), bound=bound, plane_id=plane_id)
                new_rgbs = new_rgbs.reshape(B, N, upsample_steps, 3) # [B, N, t, 3]
                new_sigmas = new_sigmas.reshape(B, N, upsample_steps) # [B, N, t]

                # re-order
                z_vals = torch.cat([z_vals, new_z_vals], dim=-1) # [B, N, T+t]
                z_vals, z_index = torch.sort(z_vals, dim=-1)

                sigmas = torch.cat([sigmas, new_sigmas], dim=-1) # [B, N, T+t]
                sigmas = torch.gather(sigmas, dim=-1, index=z_index)

                rgbs = torch.cat([rgbs, new_rgbs], dim=-2) # [B, N, T+t, 3]
                rgbs = torch.gather(rgbs, dim=-2, index=z_index.unsqueeze(-1).expand_as(rgbs))

            ### render core
            deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1] # [B, N, T-1]
            deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[:, :, :1])], dim=-1)

            alphas = 1 - torch.exp(-deltas * sigmas) # [B, N, T]
            alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-15], dim=-1) # [B, N, T+1]
            weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[:, :, :-1] # [B, N, T]

            # calculate weight_sum (mask)
            weights_sum = weights.sum(dim=-1) # [B, N]
            
            # calculate depth 
            ori_z_vals = ((z_vals - near) / (far - near)).clamp(0, 1)
            depth = torch.sum(weights * ori_z_vals, dim=-1)

            # calculate color
            image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [B, N, 3], in [0, 1]

            # mix background color
            if bg_color is None:
                bg_color = 1
                
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

            return depth, image


        def run_cuda(self, rays_o, rays_d, bound, num_steps, upsample_steps, bg_color, perturb, encoder_weights=None):
            raise NotImplementedError("run_cuda is not supported by rendererd_fibonacci yet")

        def render(self, rays_o, rays_d, bound=1, num_steps=128, upsample_steps=128, staged=False, max_ray_batch=4096, bg_color=None, perturb=False, **kwargs):
            # rays_o, rays_d: [B, N, 3], assumes B == 1
            # return: pred_rgb: [B, N, 3]

            if self.cuda_ray:
                _run = self.run_cuda
            else:
                _run = self.run

            B, N = rays_o.shape[:2]
            device = rays_o.device
            encoder_weights = kwargs['encoder_weights'] if 'encoder_weights' in kwargs else None  #pure: pass-throught encoder_weights

            # never stage when cuda_ray
            if staged and not self.cuda_ray:
                depth = torch.empty((B, N), device=device)
                image = torch.empty((B, N, 3), device=device)

                for b in range(B):
                    head = 0
                    while head < N:
                        tail = min(head + max_ray_batch, N)
                        depth_, image_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], bound, num_steps, upsample_steps, bg_color, perturb, encoder_weights=encoder_weights)
                        depth[b:b+1, head:tail] = depth_
                        image[b:b+1, head:tail] = image_
                        head += max_ray_batch
            else:
                depth, image = _run(rays_o, rays_d, bound, num_steps, upsample_steps, bg_color, perturb, encoder_weights=encoder_weights)

            results = {}
            results['depth'] = depth
            results['rgb'] = image
                
            return results