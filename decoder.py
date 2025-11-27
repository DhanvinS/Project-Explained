import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        # normalize feature before usign self attention
        self.attention = SelfAttention(1, channels)
        """Create a self-attention layer:
            1 attention head (1)
            Input dimension = number of channels
            Allows each pixel to attend to all other pixels."""
    
    def forward(self, x):

        residue = x 
        x = self.groupnorm(x)
        # save features before attention
        n, c, h, w = x.shape
        # Save batch size, channels, height, width.
        x = x.view((n, c, h * w))
        # Flatten spatial dimensions → (B, C, H*W).
        # Treat each pixel as a vector of size C.
        x = x.transpose(-1, -2)
        
        x = self.attention(x)
        # apply self attention
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        # reshape back to original
        x += residue
        # add residual i/p

        return x 

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        # Normalizes the input features
        # helps stabilize training
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # 3*3 conv layer transforms features
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        """Because two smaller convs give more power and flexibility than one big conv, and each conv benefits from its own normalize→activation step"""
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        """Shortcut connection: ensures residue has the same number of channels as output.
            If channels match → just add input as-is.
            If channels differ → use 1×1 conv to match channel count"""

        
    
    def forward(self, x):

        residue = x
        # saving input for residual connection
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        # Normalize -> activate -> conv(1st transformation)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        #  Normalize -> activate -> conv(2nd transformation)
        
        return x + self.residual_layer(residue)
        # Add original input (residue) to output → residual connection

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(

            """The decoder reconstructs the image from the latent vector z, which already encodes “what” and “where” information about the original image.
                What provides reconstruction info
                The encoder has already packed the image into z (shape like 4×64×64), where each channel and spatial location stores high-level features of the original image.
                The decoder is just a learned inverse: a stack of upsampling + conv layers trained so that, given z, it learns how to “inflate” those features back into a full 3×512×512 image that matches the input"""

            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 512), 
            
            VAE_AttentionBlock(512), 
            
            VAE_ResidualBlock(512, 512), 
            
            VAE_ResidualBlock(512, 512), 
            
            VAE_ResidualBlock(512, 512), 
            
            VAE_ResidualBlock(512, 512), 
            
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            VAE_ResidualBlock(512, 512), 
            
            VAE_ResidualBlock(512, 512), 
            
            VAE_ResidualBlock(512, 512), 
            
            nn.Upsample(scale_factor=2), 
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            VAE_ResidualBlock(512, 256), 
            
            VAE_ResidualBlock(256, 256), 
            
            VAE_ResidualBlock(256, 256), 
            
            nn.Upsample(scale_factor=2), 
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            
            VAE_ResidualBlock(256, 128), 
            
            VAE_ResidualBlock(128, 128), 
            
            VAE_ResidualBlock(128, 128), 
            """Residual blocks: allow deep transformations while preserving information (like in encoder).
                Attention block: captures long-range dependencies across spatial locations, useful for reconstructing coherent structures (e.g., faces, objects).
                Multiple blocks in a row: gradually refine the latent, combining local + global information."""
            
            nn.GroupNorm(32, 128), 
            
            nn.SiLU(), 
            
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        
        x /= 0.18215

        for module in self:
            x = module(x)

        #Pass latent through the sequential layers: residuals, attention, upsampling, convs.
        #Each step refines the features and increases resolution.

        return x