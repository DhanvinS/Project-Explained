import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)
        # explanding embeddign so model has more space to work on

    def forward(self, x):
        
        x = self.linear_1(x)
        x = F.silu(x) 
        x = self.linear_2(x)
        # apply the transformation
        return x

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        """groupnorm_feature + conv_feature = process spatial features.
           linear_time = transforms timestep embedding to match out_channels."""

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # second round of that

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            # step to make sure in channel = out channels if not it use 1 * 1 conv layer to convert
    
    def forward(self, feature, time):
        

        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        # Normalize → activate → conv = process input feature map
        
        time = F.silu(time)
        time = self.linear_time(time)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        """It turns the scalar timestep info into a vector and adds it onto every pixel’s channels.
            First two lines: make a per‑channel “time embedding” vector.​
            Last line: add that vector to the feature map everywhere, so the block “knows” which diffusion step it’s processing"""
        
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        return merged + self.residual_layer(residue)
        # combine spacial feature with timestep with residual

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        """Implements self-attention + cross-attention + feedforward (GeGLU) block in UNet.
          Used to model global interactions between pixels/features and text embedding (cross-attention)"""
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        # normalize features and prepare the mfor attention

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        """attention_1 = self-attention (pixels attend to other pixels).
           attention_2 = cross-attention (pixels attend to text embedding / context)."""
        
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        # GeGLU feedforward network for feature transformation.
        # conv_output = final linear layer after reshaping features back to (B, C, H, W)
    
    def forward(self, x, context):
        

        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        # Flatten spatial dimensions → sequence (B, H*W, C) for attention
        
        x = x.transpose(-1, -2)
        
        
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short
        # Self-attention + residual → allows pixels to communicate globally.
       
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short
        # cross attention
    
        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short
        # apply GeLU
        
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        # reshape back

        return self.conv_output(x) + residue_long

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
       
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)
    # Doubles spatial resolution of feature maps for decoder
    """This Upsample block is “zooming in” the feature map, then letting a conv clean up the zoomed image.
            Simple explanation
            First, interpolate(..., scale_factor=2) just doubles H and W (2× bigger image) using nearest‑neighbor, which is fast but a bit blocky.​
            Then the 3×3 conv learns how to smooth and refine those enlarged features so the upsampled representation is meaningful, not just a pixelated copy"""

class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
    """Loops over each layer in order.

    If the layer is an attention block, it calls layer(x, context) (needs the text/context embedding).​

    If it’s a residual block, it calls layer(x, time) (needs the time embedding).​

    Otherwise, it just does layer(x) for normal layers"""
    # Y we need this is SwitchSequential lets you build the UNet as one ordered list of
    #  layers while automatically giving each layer the right arguments, keeping the model 
    # code cleaner and easier to reason about

class UNET(nn.Module):
    # Encoder + Bottleneck + Decoder
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([

            """Multiple residual and attention blocks let the UNet do the job in small steps, at different depths and scales, instead of trying to learn everything in one huge, brittle laye"""
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            # a 2D conv layer take 4 i/p channelsand o/p 320 channels using 3*3 kernel
            # In this conv, the kernel is the small 3×3 weight window that slides over the image/feature map to compute each output pixel
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            # stride is the kernel jumps or slides 2 pixels while running over the i/p(uk this)
            
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280), 
            
            UNET_AttentionBlock(8, 160), 
            
            UNET_ResidualBlock(1280, 1280), 
        )
        """Input: deepest feature map from encoder (1280 channels, very low resolution).
        First residual block: local processing.
        Attention block: global reasoning + text prompt alignment.
        Second residual block: refine features for decoding.
        Output: enriched, globally-aware feature map ready for upsampling in the decoder"""
        
        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x
        # run the model 

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # First it normalizes the channels (GroupNorm) and applies SiLU to get clean, well‑scaled features
        """The last 3×3 conv is the readout layer: it converts rich internal features into the
          exact thing the diffusion model must predict (noise / image channels), so the loss can
            be computed"""
        
    def forward(self, x):

        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        # implements it 
        return x

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent, context, time):

        time = self.time_embedding(time)
        # Converts the scalar timestep into a 320‑dim vector so the model knows
        #  “where in the denoising process” it is
        output = self.unet(latent, context, time)
        # Feeds the noisy latent, text/context, and time embedding into the UNet to
        #  produce a 320‑channel feature map that encodes the predicted noise pattern
        
        output = self.final(output)
        # Uses the output layer to map 320 channels down to 4 latent channels, giving 
        # the final predicted noise epislon^ for this step, which the diffusion scheduler will use
        return output