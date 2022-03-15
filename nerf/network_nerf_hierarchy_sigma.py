import torch
import torch.nn as nn
from .renderer_hierarchy import NeRFRenderer
from encoding import FreqEncoder

# The Original NeRF network
class NeRFNetwork(NeRFRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(cuda_ray=kwargs['cuda_ray'])
        print("============================")
        print("Network: NeRF (Original/NeX360 Sigma mode)")
        print("CAUTION: This will use the Original NeRF. input config will be ignore")
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
        sigma_front0 = nn.Sequential(
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
        sigma_back0 = nn.Sequential(
            nn.Linear(self.in_dim + self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #from NeRF supplyment, No ReLU on last layer
        )
        sigma_layer0 = nn.Sequential(nn.Linear(self.hidden_dim, 1))
        sigma_front1 = nn.Sequential(
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
        sigma_back1 = nn.Sequential(
            nn.Linear(self.in_dim + self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #from NeRF supplyment, No ReLU on last layer
        )
        sigma_layer1 = nn.Sequential(nn.Linear(self.hidden_dim, 1))
        self.sigma_net = nn.ModuleList([sigma_front0, sigma_back0, sigma_layer0, sigma_front1, sigma_back1, sigma_layer1])

    def build_color_net(self):
        color_lvl0 = nn.Sequential(
            nn.Linear(self.in_dim_color, self.hidden_dim_color),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_color, 3),
            nn.Sigmoid()
        )
        color_lvl1 = nn.Sequential(
            nn.Linear(self.in_dim_color, self.hidden_dim_color),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_color, 3),
            nn.Sigmoid()
        )
        self.color_net = nn.ModuleList([color_lvl0, color_lvl1])


    
    def forward(self, x, d, network_level=1, *args, **kwargs):
        x_ = self.encoder(x)
        h = self.sigma_net[(3*network_level) + 0](x_)
        h = self.sigma_net[(3*network_level) + 1](torch.cat([h,x_],dim=-1))
        sigma = torch.abs(self.sigma_net[(3*network_level) + 2](h))
        d_ = self.encoder_dir(d)
        color = self.color_net[network_level](torch.cat([h,d_],dim=-1))
        return sigma, color

        
    def density(self, x, *args,  **kwargs):
        x_ = self.encoder(x)
        h = self.sigma_net[3](x_)
        h = self.sigma_net[4](torch.cat([h,x_],dim=-1))
        sigma = torch.abs(self.sigma_net[5](h))
        return sigma

    