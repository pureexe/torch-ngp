import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 log2_hashmap_size=19,
                 cuda_ray=False,
                 ):
        super().__init__(cuda_ray)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        self.encoders = []
        self.n_levels = 16
        self.n_features_per_level = 2
        self.n_projectors = 3
        for i in range(self.n_projectors):
            self.encoders.append(
                tcnn.Encoding(
                n_input_dims=2,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": self.n_levels,
                    "n_features_per_level": self.n_features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": 16,
                    "per_level_scale":  1.3819,
                },
              )
            )
    
        self.sigma_net = tcnn.Network(
            n_input_dims=(self.n_levels * self.n_features_per_level) * len(self.encoders),
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )
        #self.encoder = torch.nn.ParameterList(self.encoders) #require to make save & load weight work

    def projected_to_plane(self, x):
        # x: [B,N,3] contain position x and y and z
        projected = [] #projected cotain list of projected value to 2d plane

        # project to xy plane
        projected.append(x[...,:2])
        # project to xz plane
        projected.append(x[...,0:3:2])
        # projected to yz plane
        projected.append(x[...,1:])

        return projected

    def geo_network(self, x):
        projected = self.projected_to_plane(x)
        features = [] 
        for i,p in enumerate(projected):
            features.append(self.encoders[i](p))
        features = torch.cat(features, dim=-1)
        h = self.sigma_net(features)
        return h


    def forward(self, x, d, bound):
        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]

        prefix = x.shape[:-1]
        x = x.view(-1, 3)
        d = d.view(-1, 3)

        # sigma
        x = (x + bound) / (2 * bound) # to [0, 1]

        h = self.geo_network(x)
        #x = self.encoder(x)
        #h = self.sigma_net(x)

        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

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

    def density(self, x, bound):
        # x: [B, N, 3], in [-bound, bound]

        prefix = x.shape[:-1]
        x = x.view(-1, 3)

        x = (x + bound) / (2 * bound) # to [0, 1]
        
        h = self.geo_network(x)
        #x = self.encoder(x)
        #h = self.sigma_net(x)

        #sigma = torch.exp(torch.clamp(h[..., 0], -15, 15))
        sigma = F.relu(h[..., 0])

        sigma = sigma.view(*prefix)

        return sigma