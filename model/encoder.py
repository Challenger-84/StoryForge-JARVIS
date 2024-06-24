import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (B, C=3, H, W) --> (B, C=128, H, W)
            nn.conv2d(3, 128, kernel_size=3, padding=1),

            # (B, 128, H, W) --> (B,128, H, W)
            VAE_ResidualBlock(128, 128),
            # (B, 128, H, W) --> (B,128, H, W)
            VAE_ResidualBlock(128, 128),
            # (B,128, H, W) --> (B,128, H/2, W/2)
            nn.conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (B,128, H/2, W/2) --> (B,256, H/2, W/2)
            VAE_ResidualBlock(128, 256),
            # (B,256, H/2, W/2) --> (B,256, H/2, W/2)
            VAE_ResidualBlock(256, 256),
            # (B,256, H/2, W/2) --> (B,256, H/4, W/4)
            nn.conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (B,256, H/4, W/4) --> (B,512, H/4, W/4)
            VAE_ResidualBlock(256, 512),
            # (B,512, H/4, W/4) --> (B,512, H/4, W/4)
            VAE_ResidualBlock(512, 512),
            # (B,512, H/4, W/4) --> (B,512, H/8, W/8)
            nn.conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (B,512, H/8, W/8) --> (B,512, H/8, W/8)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512), # no change in shape
            VAE_ResidualBlock(512, 512),# no change in shape
            nn.GroupNorm(32, 512),# no change in shape
            nn.SiLU(),# no change in shape

            # (B, 512, H/8, W/8) --> (B, 8, H/8, W/8)
            nn.conv2d(512, 8, kernel_size=3, padding=1),
            # (B, 8, H/8, W/8) --> (B, 8, H/8, W/8)
            nn.conv2d(8, 8, kernel_size=1, padding=0),

            # So final shape is (Batch_size, Channel = 8, H/8, W/8)
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x : (B, 3, H, W)
        # noise : (B, 8, H/8, W/8)
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (pad_left, pad_right, pad_top, pad_bottom)
                x = F.pad(x, (0, 1, 0, 1)) # apply padding at top and bottom only
            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=2) # (B, 8, H/8, W/8) --> two tensors of shape (B, 4, H/8, W/8)

        log_variance = torch.clamp(log_variance, -30, 20)# no change in shape
        variance = log_variance.exp()# no change in shape
        stdev = variance.sqrt()# no change in shape

        x *= 0.18215
        
        return x