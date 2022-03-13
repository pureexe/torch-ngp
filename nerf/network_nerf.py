import torch
import torch.nn as nn
from .renderer import NeRFRenderer
from encoding import FreqEncoder

# The Original NeRF network
class NeRFNetwork(NeRFRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(cuda_ray=kwargs['cuda_ray'])
        super
        print("============================")
        print("Network: NeRF (Original)")
        print("CAUTION: This will use the Original NeRF")
        print("Number of layer will set to 8 for sigma and 1 for color")
        print("Encoder will set to Frequncy")
        print("and other input config will be ignore")
        print("============================")
    
        # need to make a config publicly accessible by Trainer/Renderer
        self.num_layers = 8 
        self.hidden_dim = 256 
        self.num_layers_color = 1        
        self.hidden_dim_color = 128
        self.geo_feat_dim = self.hidden_dim 
        self.encoder = FreqEncoder(input_dim=3, max_freq_log2=9, N_freqs=10)
        self.encoder_dir = FreqEncoder(input_dim=3, max_freq_log2=3, N_freqs=4)
        self.in_dim = self.encoder.output_dim
        self.in_dim_color = self.encoder_dir.output_dim + self.geo_feat_dim
        
        self.build_sigma_net()
        self.build_color_net()

    def build_sigma_net(self):
        # build 8 layers network
        sigma_front = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        sigma_back = nn.Sequential(
            nn.Linear(self.in_dim + self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim+1), #from NeRF supplyment, No ReLU on last layer
        )
        self.sigma_net = nn.ModuleList([sigma_front, sigma_back])

    def build_color_net(self):
        layers = [
            nn.Linear(self.in_dim_color, self.hidden_dim_color),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_color, 3),
            nn.Sigmoid()
        ]
        self.color_net = nn.Sequential(*layers)


    def forward(self, x, d, *args, **kwargs):
        x_ = self.encoder(x)
        h = self.sigma_net[0](x_)
        h = self.sigma_net[1](torch.cat([h,x_],dim=-1))
        sigma = torch.relu(h[...,:1])
        h_ = h[...,1:]
        d_ = self.encoder_dir(d)
        color = self.color_net(torch.cat([h_,d_],dim=-1))
        return sigma, color

        
    def density(self, x, *args,  **kwargs):
        x_ = self.encoder(x)
        h = self.sigma_net[0](x_)
        h = self.sigma_net[1](torch.cat([h,x_],dim=-1))
        sigma = torch.relu(h[...,:1])
        return sigma