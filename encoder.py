import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            """All these are initialized inside init In PyTorch, the __init__ method of a nn.Module is 
                where you declare all the layers your model will use.
                These layers are created once and then reused every time you call forward().
                Think of it as building the blueprint of your encoder"""
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # The layer learns 128 different filters (one per output channel), each detecting
            #  a different feature like edges or textures, so you go from 3 channels (RGB) to 128
            #  feature maps while keeping the image height and width the same
            
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            # Residual blocks help the network learn complex mappings without vanishing gradients.
            # No downsampling here; shape stays (Batch, 128, H, W)
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            """Downsampling layer: reduces spatial dimensions H × W → H/2 × W/2
            This is compression in the VAE sense: the image is represented in a smaller latent space.
            Analogy: zooming out, keeping key information, discarding high-resolution detai"""

            
            VAE_ResidualBlock(128, 256), 
            VAE_ResidualBlock(256, 256), 
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 

            
            VAE_ResidualBlock(256, 512),    
            VAE_ResidualBlock(512, 512), 
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            """Multiple residual blocks let the network do many small, safe refinements
              instead of one huge, hard transformation, while keeping gradients flowing well"""
            
            VAE_AttentionBlock(512), 
            # Attention allows long-range interactions across the latent feature map.
            # Important because a pixel in one corner might relate to a pixel far away.
            # In Stable Diffusion: this helps the latent capture global image context

            VAE_ResidualBlock(512, 512), 
            
            nn.GroupNorm(32, 512), 
            """Normalizing makes the activations inside the network behave like a standardized 
            distribution (roughly zero mean, unit variance) so they are on a stable, predictable
            scale for the next layers.​
            This “stable” distribution makes training smoother: gradients don’t explode/vanish
            as easily, and the model can learn better representations because each layer sees
            inputs with similar statistics instead of wild, drifting values"""

            nn.SiLU(), 
            """nn.GroupNorm(32, 512) normalizes the 512 channels by splitting them into 32 
            groups of 16 channels each and computing mean/std inside each group, which stabilizes
            training and is more robust than BatchNorm when batch sizes are small.​
            nn.SiLU() is an activation function that applies silu
            silu(x)=x⋅sigmoid(x) element‑wise, giving a smooth, ReLU‑like nonlinearity that 
            keeps small negative gradients instead of zeroing them out"""

            nn.Conv2d(512, 8, kernel_size=3, padding=1), 
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
            """Final conv layers reduce channels down to 8.
            Why 8? Because later we split into two tensors of 4 channels each: μ and log σ².
            Analogy: the network outputs parameters of the Gaussian that represents the latent distribution of the input image"""
        )

    def forward(self, x, noise):
        # x = input image (B, 3, H, W)
        # noise = standard Gaussian noise (B, 4, H/8, W/8) used for reparameterization
        # This connects directly to what we discussed: sampling z = μ + σ·ε

        for module in self:
            # Loop over each sub-module inside self (e.g., each layer in a nn.Sequential)
            if getattr(module, 'stride', None) == (2, 2): 
                # Check if this module has an attribute called stride and that it equals
                x = F.pad(x, (0, 1, 0, 1)) 
                # If it is stride‑2, pad x on the right and bottom by 1 pixel: (left, right, top, bottom) = (0, 1, 0, 1), so width and height each get 1 extra pixel only on one side
            
            x = module(x)


        mean, log_variance = torch.chunk(x, 2, dim=1)
        """Splits the final output into μ (mean) and log σ² (log variance).
            Each has 4 channels: (B, 4, H/8, W/8)
            Connection to theory: this is the probabilistic encoding. We now have a Gaussian distribution for each latent pixel, not a single point"""
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        """Clamp to avoid numerical instability (σ can’t be too small or too big).
           Clamping just means forcing values to stay inside a chosen range by cutting off anything too low or too high
            Convert log variance → variance → standard deviation (σ)
            Connection: this is exactly the σ in z = μ + σ·ε we’ve been discussing."""
        
        x = mean + stdev * noise
        # reparameterization
        
        x *= 0.18215
        """x *= 0.18215 scales the latent so its values match what the UNet expects.
        Without it, the latent would be too large and the network would produce bad images.
        The number 0.18215 was chosen during training so the latent has roughly unit variance.
        Analogy: it’s like resizing clay to the right thickness before sculpting."""
        