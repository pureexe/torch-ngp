import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn
from .renderer_fibonacci import NeRFRenderer
from .nex360 import get_uniform_sphere

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
                 fibonacci=None,
                 blend_lattice=1,
                 global_tri=False,
                 **kwargs
                 ):
                 
        super().__init__(cuda_ray=cuda_ray)

        if encoding != "HashGrid" and encoding != "hashgrid" :
            raise NotImplementedError("Sigma Network currently supprot only HashGrid")

        if encoding_dir != "SphericalHarmonics" and encoding_dir != "sphere_harmonics":
            raise NotImplementedError("Color Network currently supprot only SphericalHarmonics")

        if fibonacci == None:
            raise Exception("require to specify 'fibonacci' to be a number of encoder")

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        self.encoders = []
        self.n_levels = 16
        self.n_features_per_level = 2
        self.n_projectors = fibonacci
        self.global_tri = global_tri
        num_encoder = fibonacci if not global_tri else fibonacci + 3

        self.plane_normal = torch.from_numpy(get_uniform_sphere(num = fibonacci, is_hemi = True, radius = 1.0))[...,:3,3].cuda()
        if self.global_tri:
            self.plane_normal = torch.cat([self.plane_normal, torch.eye(3, dtype=self.plane_normal.dtype).cuda()],dim=0)

        self.projected_vector = self.precompute_projected_vector()
        

        for i in range(num_encoder):
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

        self.encoder = torch.nn.ModuleList(self.encoders) #require to make save & load weight work

        plane_per_fetch = 1 if not self.global_tri else 4
        self.sigma_net = tcnn.Network(
            n_input_dims=(self.n_levels * self.n_features_per_level) * plane_per_fetch,
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

    def precompute_projected_vector(self):
        """
        projected vector use to project point on 3d to 2d plane
        @see https://stackoverflow.com/a/23472188/5496210
        @return projected shape[...,2,3] where 2 is channel x and y
        """
        n = self.plane_normal
        x = torch.tensor([1.0,0.0,0.0]).float().cuda()[None].expand(n.shape[0],-1)
        x = x - (x[..., None, :3] @ n[...,None])[0] * n
        x = x / torch.norm(x,dim=-1)[...,None]
        y = torch.cross(n, x, dim=-1)
        projected = torch.cat([x[...,None,:],y[...,None,:]],dim=-2)
        print(projected)
        exit()
        return projected

    def projected_to_plane(self, a, plane_id):
        """
        project point "a" to plane. 
        Assume, plane origin at [0,0,0] (center of the scene)
        @see https://stackoverflow.com/a/23472188/5496210
        @params a - point in 3d space shape[B, rays,3]
        @params plane_id - id of the plane to project. note that last 3 planes might reserve for global representation
        @return p - a point on the plane shape[B, rays,2]
        """
        # get normal
        num_rays = a.shape[1]
        n = self.plane_normal[plane_id] #shape:[B,3]
        # othogonal projection
        b = a - (a[..., None, :3] @ n[...,None])[...,0].expand(-1,-1,3) * n #shape:[B, rays,3]
        projected_vector = self.projected_vector[plane_id] #shape:[B,2,3]
        x = (projected_vector[:,0:1,None,:].expand(-1,num_rays,-1, -1) @ b[...,None])[...,0]
        y = (projected_vector[:,1:2,None,:].expand(-1,num_rays,-1, -1) @ b[...,None])[...,0]
        p = torch.cat([x,y],dim=-1)
        return p

    """
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
    """


    def get_geo_feature(self, x, plane_id):
        plane_id = plane_id.long()
        projected = self.projected_to_plane(x, plane_id)
        # different batch can be a different plane, here we have to manually loop over the batch
        feature = []
        for i in range(plane_id.shape[0]):
            feature.append(self.encoders[plane_id[i]](projected[i])[None])
        features = [torch.cat(feature,dim=0)]
        if self.global_tri:
            prefix = x.shape[:-1]
            for i in range(3):
                plane_id = torch.tensor([self.n_projectors+i]).cuda().expand(x.shape[0])
                projected = self.projected_to_plane(x, plane_id)
                feature = self.encoders[plane_id](projected.view(-1, 2)) 
                feature = feature.view(*prefix,-1)
                features.append(feature)
        features = torch.cat(features, dim=-1)
        return features


    def forward(self, x, d, bound, plane_id, **kwargs):

        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]
        # plane_id: [B], contain number from in [0, self.n_projectors]

        assert plane_id is not None 
        assert len(plane_id.shape) == 1


        prefix = x.shape[:-1]
        
        #x = x.view(-1, 3)
        
        # sigma
        x = (x + bound) / (2 * bound) # to [0, 1]
        


        features = self.get_geo_feature(x, plane_id)

        # flatten features to support tiny-cuda-nn
        features = features.view(-1, self.sigma_net.n_input_dims)

        h = self.sigma_net(features)

        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

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

    def density(self, *args, **kwargs):
        raise NotImplementedError("density currently is not supprot by network_fibonacci") 
