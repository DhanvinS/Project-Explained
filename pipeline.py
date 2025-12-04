import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
# DDPMSampler – handles the diffusion process (noise addition/removal).
# torch, numpy, tqdm – standard libraries for tensor ops, array manipulations, and progress bars

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    
    """Main image generation function.

Arguments:

prompt – text prompt describing the image.

uncond_prompt – optional unconditional prompt for classifier-free guidance (CFG).

input_image – optional image for image-to-image generation.

strength – how much noise to add when using input_image.

do_cfg – whether to use classifier-free guidance.

cfg_scale – guidance strength for CFG.

sampler_name – type of sampler (here only "ddpm" supported).

n_inference_steps – number of diffusion steps.

models – dictionary with loaded encoder, decoder, diffusion, clip.

seed – for reproducibility.

device – compute device.

idle_device – device to offload unused models.

tokenizer – for tokenizing text prompts"""

     # Wraps everything in torch.no_grad() because we aren’t training; saves memory.
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")
            """That snippet is doing two things: turning off gradient tracking for the block, and checking that strength
            is in the range (0, 1], otherwise throwing a nice error"""

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
            """Checks that strength is valid.
            Defines a helper to_idle() to move unused models to idle_device (e.g., CPU)
              to save GPU memory """

        # Sets up a deterministic random number generator for sampling noise.
        # Ensures outputs are reproducible if a seed is provided
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
            
        

        clip = models["clip"]
        clip.to(device)
        # load clip
        
        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
        to_idle(clip) # move slip to idle devise
        """Converts prompt(s) into token IDs, pads them to length 77 (CLIP’s max sequence length).

        Sends them through CLIP to get context embeddings.

        If do_cfg (classifier-free guidance):

            Compute embeddings for conditional and unconditional prompts.

            Concatenate them along batch dimension → 2 * Batch_Size for later CFG processing"""



        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")
        # Instantiates DDPM sampler and sets the number of diffusion steps

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        """Shape of latent tensor:
            Batch size = 1
            Channels = 4 (latent channels for VAE)
            Height & Width = downsampled by 8"""

       # image to image option not for us
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

            """If input_image is provided:

                Resize and convert image to tensor.

                Normalize to [-1,1].

                Add batch and channel dimensions.

                Generate random noise of the same latent shape.

                Encode the image into latent space using the VAE encoder.

                Add controlled noise based on strength (for image-to-image guidance).

                Otherwise, start with pure random latent noise (text-to-image generation)."""

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents

            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(timestep, latents, model_output)
            """This loop is turning noise into an image latent step by step.

                At each timestep, it feeds the current latent + prompt into the diffusion UNet to predict the noise in that latent (with CFG combining cond/uncond if enabled).​

                The sampler then uses that predicted noise to slightly update the latent toward a cleaner one, repeating until you get a final latent that can be decoded into an image"""
            

        to_idle(diffusion)
        # idle diffusion model

        decoder = models["decoder"]
        decoder.to(device)
        images = decoder(latents)
        to_idle(decoder)


        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
        """ Send final latent through VAE decoder → reconstructs image.

            Rescale from [-1,1] to [0,255] and clamp.

            Convert tensor from (B, C, H, W) → (H, W, C) for standard image format.

            Return first image."""
    

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x
    # Linearly maps values from old_range to new_range.
    # Optionally clamps the output

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    """Implements sinusoidal time embeddings for diffusion steps (like in Transformers).

    Output shape: (1, 320) → concatenated [cos, sin] of size 160 each.

    Helps the UNet understand which diffusion timestep it’s at."""
