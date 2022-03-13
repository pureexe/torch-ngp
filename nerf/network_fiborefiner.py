import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn
from .renderer_fibonacci import NeRFRenderer
from .network_fibonacci import NeRFNetwork as FibonacciNetwork
from .nex360 import get_uniform_sphere

class NeRFNetwork(FibonacciNetwork):
    def __init__(self,
                 refiner_ratio=0.1,
                 *args,
                 **kwargs
                 ):
                 
        super().__init__(*args, **kwargs, refiner_ratio=refiner_ratio)
        if kwargs['global_tri'] == False or kwargs['global_tri'] == None:
            raise Exception("global_tri must be enable in 'fiborefiner'")
        self.refiner_ratio = refiner_ratio
        self.refine_net = (tcnn.Network(
            n_input_dims=(self.n_levels * self.n_features_per_level) + 1 + self.geo_feat_dim,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim,
                "n_hidden_layers": self.num_layers - 1
            },
        ))


    def get_tri_feature(self, x):
        features = []
        prefix = x.shape[:-1]
        for i in range(3):
            plane_id = torch.tensor([self.n_projectors+i]).long().to(x.device).expand(x.shape[0])
            projected = self.projected_to_plane(x, plane_id)
            feature = self.encoders[plane_id](projected.view(-1, 2)) 
            feature = feature.view(*prefix,-1)
            features.append(feature)
        features = torch.cat(features, dim=-1)
        return features

    def get_plane_feature(self, x, plane_id):
        plane_id = plane_id.long()
        projected = self.projected_to_plane(x, plane_id)
        feature = []
        for i in range(plane_id.shape[0]):
            feature.append(self.encoders[plane_id[i]](projected[i])[None])
        return torch.cat(feature,dim=0)


    def forward(self, x, d, bound, plane_id, **kwargs):

        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]
        # plane_id: [B], contain number from in [0, self.n_projectors]

        prefix = x.shape[:-1]
        if len(x.shape) == 2:
            x = x[None] #need to provide batch dimension
        #x = x.view(-1, 3)
        
        # sigma
        x = (x + bound) / (2 * bound) # to [0, 1]
        
        # sigma network
        features = self.get_tri_feature(x)
        features = features.view(-1, self.sigma_net.n_input_dims) # flatten features to support tiny-cuda-nn
        h = self.sigma_net(features)
        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

        # refine network
        plane_feat = self.get_plane_feature(x, plane_id).view(-1,self.n_levels * self.n_features_per_level)
        features = torch.cat([plane_feat, sigma[...,None], geo_feat], dim=-1)
        h2 = self.refine_net(features)
        sigma = sigma + self.refiner_ratio * F.tanh(h2[...,0])
        geo_feat = h2[...,1:]

        # flatten view direction to support tiny-cuda-nn
        d = d.view(-1, 3)
        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        #p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)
    
        sigma = sigma.view(*prefix)
        color = color.view(*prefix, -1)

        return sigma, color

    def density(self, x, bound, **kwargs):

        prefix = x.shape[:-1]
        if len(x.shape) == 2:
            x = x[None] #need to provide batch dimension

        x = (x + bound) / (2 * bound) # to [0, 1]        
    
        # sigma network
        features = self.get_tri_feature(x)
        features = features.view(-1, self.sigma_net.n_input_dims) # flatten features to support tiny-cuda-nn
        h = self.sigma_net(features)
        sigma = F.relu(h[..., 0])
        
        return sigma.view(*prefix)
