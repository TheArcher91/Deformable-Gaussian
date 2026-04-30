import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs):
        super().__init__()
        self.num_freqs = num_freqs
        # Create frequencies: 2^0, 2^1, ..., 2^(L-1)
        self.register_buffer("freqs", 2.0 ** torch.arange(num_freqs))

    def forward(self, x):
        """
        Inputs: x of shape [N, D]
        Outputs: encoded x of shape [N, D + D * 2 * num_freqs]
        """
        out = [x]
        for freq in self.freqs:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)

class DeformationField(nn.Module):
    # DOWNSIZED FOR 6GB VRAM LIMITS
    def __init__(self, W=64, spatial_freqs=4, temporal_freqs=2):
        super().__init__()
        
        # 1. Initialize the Encoders
        self.pe_spatial = PositionalEncoding(spatial_freqs)
        self.pe_temporal = PositionalEncoding(temporal_freqs)
        
        # 2. Calculate the new input dimensions
        spatial_dim = 3 + (3 * 2 * spatial_freqs)
        temporal_dim = 1 + (1 * 2 * temporal_freqs)
        in_dim = spatial_dim + temporal_dim
        
        # 3. The Downsized MLP
        # Dropping W from 128 to 64 cuts the internal memory footprint by 75%
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU() # Removed one entire hidden layer
        )
        
        self.delta_pos = nn.Linear(W, 3)    
        self.delta_scale = nn.Linear(W, 3)  
        self.delta_rot = nn.Linear(W, 4)    
        
        self._initialize_static()

    def _initialize_static(self):
        # Starts the network as perfectly static
        for module in [self.delta_pos, self.delta_scale, self.delta_rot]:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, positions, time):
        # 4. Encode the inputs before passing them to the MLP
        encoded_positions = self.pe_spatial(positions)
        encoded_time = self.pe_temporal(time)
        
        x = torch.cat([encoded_positions, encoded_time], dim=-1)
        features = self.mlp(x)
        
        d_pos = self.delta_pos(features)
        d_scale = self.delta_scale(features)
        d_rot = self.delta_rot(features)
        
        return d_pos, d_scale, d_rot