"""
Train the diffusion UNet on Flickr30k captions + images,
using pretrained CLIP and VAE weights from your existing repo.

Save this as: train_flickr30k_unet.py

"""

import os
import math
import random
from dataclasses import dataclass
# import python dataclass 
# Here it is used to define a training config class (TrainConfig) that just holds settings (paths, hyperparameters) in a clean, typed way.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# They import PyTorch’s dataset abstraction and batching/loader utility

from datasets import load_dataset
# load_dataset: HuggingFace datasets to fetch nlphuji/flickr30k.
from transformers import CLIPTokenizer
# CLIPTokenizer is just for turning text into token IDs compatible with a CLIP text encoder; it has no neural network weights by itself.
from tqdm import tqdm
# tqdm: progress bar for loops.
# Progress‑bar utility to wrap loops (e.g., training or sampling) and show live progress in the terminal.

import model_loader  # from your repo

# -----------------------------
# Config
# -----------------------------

WIDTH = 512
HEIGHT = 512
# Your final images are 512x512.

LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8
# Latent spatial size is image_size / 8 (standard SD VAE downsample factor).

MAX_TEXT_LENGTH = 77        
# CLIP tokenizer sequence length
# 77 matches CLIP’s sequence length.

NUM_TRAIN_TIMESTEPS = 1000 
# DDPM training steps


@dataclass
class TrainConfig:
    vocab_path: str = "data/tokenizer_vocab.json"    # ### TODO: adjust if paths differ
    merges_path: str = "data/tokenizer_merges.txt"   # ### TODO: adjust if paths differ
    model_ckpt_path: str = "data/v1-5-pruned-emaonly.ckpt"  # ### TODO: your SD checkpoint
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    num_epochs: int = 1
    learning_rate: float = 1e-5
    num_train_timesteps: int = NUM_TRAIN_TIMESTEPS
    num_workers: int = 2
    output_dir: str = "checkpoints"
    max_train_steps_per_epoch: int | None = None  # set to int to limit per-epoch steps


"""TrainConfig is a dataclass holding all training hyperparameters and paths:

vocab_path, merges_path: files for CLIP tokenizer.

model_ckpt_path: your Stable Diffusion checkpoint.

device: use GPU if available.

batch_size, num_epochs, learning_rate: training hyperparameters.

num_train_timesteps: noise schedule length.

num_workers: number of dataloader workers.

output_dir: where checkpoints go.

max_train_steps_per_epoch: optional limit if you don’t want to go through entire dataset each epoch"""
# -----------------------------
# Utility: Rescale + time embedding
# -----------------------------

def rescale(x: torch.Tensor, old_range, new_range, clamp: bool = False) -> torch.Tensor:
    """
    Same rescale function as in your pipeline:
    maps from old_range to new_range linearly.
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    x = x.clone()
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

"""This linearly maps tensor x from [old_min, old_max] to [new_min, new_max].

Used for mapping image pixels [0,255] -> [-1,1] or latents back, same as your generation code.

x.clone() so you don’t modify original tensor in-place.

clamp=True ensures values stay in the target range."""
# each epoch goes through the entire flikr dataset
# and each image is used to train with multiple timesteps not just one 


def get_time_embedding(timesteps: torch.LongTensor) -> torch.Tensor:
    """
    Vectorised version of your get_time_embedding for a batch of timesteps.
    Input: timesteps (B,) long
    Output: (B, 320) time embeddings
    """
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (B, 1)
    x = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B,160)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)  # (B,320)

    # It’s turning the scalar timestep into a 320‑dim numeric code using many cos/sin waves with different frequencies
    # what this is used for - That embedding is used to tell the U‑Net which diffusion step it is currently at, 
    # so it can denoise differently at early vs late steps
    """Turning t into 320 dims gives the network a rich, multi‑scale signal about “where in the process we are”, instead of a single raw number."""
    


        """This is the positional / time embedding used by your UNet:

        Similar to Transformer sinusoidal embeddings.

        freqs: 160 frequency values.

        x: shape (B, 160) = timesteps scaled by those frequencies.

        Returns [cos(x), sin(x)] concatenated → (B, 320).

        Matches your single-timestep get_time_embedding, but vectorized for a batch."""


# -----------------------------
# Flickr30k Dataset wrapper
# -----------------------------

class Flickr30kDiffusionDataset(Dataset):
    """
    Wraps HF dataset `nlphuji/flickr30k` into a PyTorch Dataset that returns:
      - pixel_values: (3, H, W) in [-1, 1]
      - input_ids: (MAX_TEXT_LENGTH,)
    """
    # Custom Dataset class for Flickr30k → ready for diffusion training.

    def __init__(self, hf_split, tokenizer: CLIPTokenizer, image_size: int = 512, max_length: int = 77):
        self.data = hf_split
        self.tokenizer = tokenizer
        # Store the provided CLIPTokenizer instance, to be used later to tokenize captions
        self.image_size = image_size
        # Remember the desired image resolution (e.g. 512) so each image can be resized to this shape when loaded
        self.max_length = max_length
        """Remember the maximum text length (e.g. 77 tokens) so tokenization can pad/truncate captions consistently for CLIP"""
        # hf_split: the Hugging Face dataset split 
        # Store tokenizer and some hyperparameters.

    def __len__(self):
        return len(self.data)
    # Number of samples is just the length of HF split

    def __getitem__(self, idx):
        example = self.data[idx]
        """Get one example from HF dataset"""
        # ----- Image preprocessing -----
        
        image = example["image"].resize((self.image_size, self.image_size))
        # Resize the PIL image to 512 x 512.
        """PIL (Python Imaging Library), now maintained as Pillow, adds support for opening, processing, and saving many image formats like PNG and JPEG in Python."""

        import numpy as np
        image = np.array(image).astype("float32")
        image = torch.from_numpy(image)  
        """Convert PIL → numpy array → torch tensor with shape (H, W, C)."""
        
        image = rescale(image, (0, 255), (-1, 1))
        # 4) (H,W,C) -> (C,H,W)
        """Use your rescale to match the scale used by your VAE / pipeline."""

        image = image.permute(2, 0, 1)
        # Change to channel-first format (C, H, W) for PyTorch / VAE.
        # height width of image and channel

        # ----- Caption preprocessing -----
        # Flickr30k has multiple captions per image in "raw". Take a random one.
        captions = example["raw"]
        caption = random.choice(captions)
        # Each image has multiple captions; pick one randomly to add some variation.

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)  # (77,)
        """Tokenize with CLIP tokenizer:

            padding="max_length" → always 77 tokens.

            truncation=True → cut longer captions. if the lenth is longer than max length cut offextra tokens

            return_tensors="pt" → PyTorch tensors.

            input_ids shape is (77,)."""


        return {
            "pixel_values": image,   # (3,H,W), float32 in [-1,1]
            "input_ids": input_ids,  # (77,), long
        }
    
    """Return dict for this sample: image and tokenized caption."""


def create_dataloader(cfg: TrainConfig, tokenizer: CLIPTokenizer) -> DataLoader:

    ds = load_dataset("nlphuji/flickr30k")
    # Downloads and loads nlphuji/flickr30k from HF.

    # We'll just use train split here; you can add val if you want.
    hf_train = ds["train"]

    train_dataset = Flickr30kDiffusionDataset(
        hf_split=hf_train,
        tokenizer=tokenizer,
        image_size=WIDTH,
        max_length=MAX_TEXT_LENGTH,
    )
    # Wrap HF split into your custom dataset with your sizes.

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader
    # Takes your dataset and automatically groups samples into batches (e.g., 32 images + captions at a time).​
    # Lets you shuffle and load data efficiently while you loop over it in your training loop

    """Creates PyTorch DataLoader:

    shuffle=True: randomize order each epoch.

    num_workers: multi-process loading.

    pin_memory=True: better GPU transfer.

    drop_last=True: ensures all batches have equal batch_size."""


# -----------------------------
# Noise schedule for DDPM training
# -----------------------------

class NoiseScheduler:
    """
    A simple DDPM-style noise scheduler, independent of your DDPMSampler.
    This is used only for training (q(x_t | x_0)).
    """
    # Implements q(x_t | x_0) to generate noisy latents for training.

    def __init__(self, num_train_timesteps: int, device: str):
        self.num_train_timesteps = num_train_timesteps
        self.device = device
        # Stores number of timesteps and device.

        beta_start = 0.00085
        beta_end = 0.012
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # Defines a linear schedule for betas from small to larger noise amounts.

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        """alphas = 1 - beta.

           alphas_cumprod[t] = Π_{i=0}^t α_i, standard DDPM cumulative.     """
        

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        """Precompute:

                sqrt(α̅_t) and sqrt(1-α̅_t) for all timesteps — needed for q_sample"""


    def q_sample(self, x0: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Draw x_t ~ q(x_t | x_0, t)
        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * noise
        """
        # t: (B,)
        # gather per-sample coefficients
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise
        
    """Given:

        x0: clean latents,

        t: timesteps for each sample,

        noise: sampled ε ~ N(0, I),

        Returns x_t, the noisy latents, using DDPM formula."""




# -----------------------------
# Model loading
# -----------------------------

def load_models_and_tokenizer(cfg: TrainConfig):
    """
    Load tokenizer and models (clip, encoder, diffusion, decoder) using your existing code.
    """
     # Central function to load pretrained stuff.

    # CLIP tokenizer (same as your generate.py)
    tokenizer = CLIPTokenizer(cfg.vocab_path, merges_file=cfg.merges_path)

    # Pretrained SD-like weights
    models = model_loader.preload_models_from_standard_weights(cfg.model_ckpt_path, cfg.device)
    """Use your model_loader helper to load models from SD checkpoint.

        models should be a dict with "clip", "encoder", "diffusion", "decoder"."""

    clip = models["clip"]
    encoder = models["encoder"]
    diffusion = models["diffusion"]
    decoder = models["decoder"]
    # Extract individual components.

    # Move to device
    clip.to(cfg.device)
    encoder.to(cfg.device)
    diffusion.to(cfg.device)
    decoder.to(cfg.device)
    """Put all models on GPU or CPU as per config."""

    # Freeze CLIP and VAE
    for p in clip.parameters():
        p.requires_grad = False
    for p in encoder.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False
        """Turn off gradients for CLIP and VAE encoder/decoder → they won't be updated."""

    clip.eval()
    encoder.eval()
    decoder.eval()
    diffusion.train()  # we're training only the UNet
    """Set modes:

clip, encoder, decoder in eval mode.

diffusion in train mode (UNet will be updated)."""

    return models, tokenizer


# -----------------------------
# Training loop
# -----------------------------

def train_unet_on_flickr30k(cfg: TrainConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
     # Create directory for checkpoints if it doesn’t exist.

    print(f"Using device: {cfg.device}")
    print("Loading models and tokenizer...")
    models, tokenizer = load_models_and_tokenizer(cfg)
    # Logs and loads models/tokenizer.

    clip = models["clip"]
    encoder = models["encoder"]
    diffusion = models["diffusion"]
    decoder = models["decoder"]  # not used in training, but loaded
    """For convenience, unpack the dict."""

    print("Creating dataloader...")
    train_loader = create_dataloader(cfg, tokenizer)
    """Build the PyTorch dataloader from Flickr30k."""

    print("Building noise scheduler...")
    noise_scheduler = NoiseScheduler(cfg.num_train_timesteps, cfg.device)
    """Create the DDPM noise scheduler for q(x_t | x_0)."""

    # Optimizer only on diffusion (UNet) params
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=cfg.learning_rate)
    """Use AdamW optimizer on only diffusion UNet parameters.

        CLIP and VAE are frozen so their params are not updated."""

    global_step = 0
    # Track how many steps we've trained globally.

    for epoch in range(cfg.num_epochs):
        diffusion.train()
        running_loss = 0.0
        """Loop over epochs.

            Ensure diffusion is in train() mode.

            Reset running_loss."""

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", leave=True)
        for step, batch in enumerate(pbar):
            if cfg.max_train_steps_per_epoch is not None and step >= cfg.max_train_steps_per_epoch:
                break
            """Wrap dataloader with a progress bar.

                Optional early break per epoch if max_train_steps_per_epoch is set."""


            pixel_values = batch["pixel_values"].to(cfg.device)  # (B,3,H,W), [-1,1]
            input_ids = batch["input_ids"].to(cfg.device)        # (B,77)
            # Move images and token IDs to GPU/CPU.

            batch_size = pixel_values.shape[0]
            # Get current batch size.

            # -------------------------
            # 1. Encode text with CLIP (frozen)
            # -------------------------
            with torch.no_grad():
                # (B,77) -> (B,77,Dim)
                context = clip(input_ids)
                """Computes CLIP text embeddings for the captions:

                input_ids shape (B, 77) → context shape (B, 77, Dim).

                no_grad because CLIP is frozen."""


            # -------------------------
            # 2. Encode image into latents with VAE encoder (frozen)
            # -------------------------
            with torch.no_grad():
                # (B,3,H,W) -> (B,3,H,W) already, but encoder expects (B,C,H,W) in [-1,1]
                encoder_noise = torch.randn(
                    (batch_size, 4, LATENTS_HEIGHT, LATENTS_WIDTH),
                    device=cfg.device,
                )
                # (B,4,LatH,LatW)
                latents = encoder(pixel_values, encoder_noise)
                """Generates random encoder_noise for the VAE’s stochastic encoding.

                Passes images + noise to encoder to get latents (B, 4, 64, 64 for 512x512).

                Again no_grad because encoder is frozen."""


            # -------------------------
            # 3. Sample random timesteps + noise, create noisy latents
            # -------------------------
            timesteps = torch.randint(
                low=0,
                high=cfg.num_train_timesteps,
                size=(batch_size,),
                device=cfg.device,
                dtype=torch.long,
            )
            """For each example in the batch, pick a random timestep t in [0, num_train_timesteps)."""


            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.q_sample(latents, timesteps, noise)
            """Sample standard Gaussian noise ε with same shape as latents.

            Use q_sample to create x_t (noisy latents) from x_0 (latents) and noise."""

            # -------------------------
            # 4. Time embeddings
            # -------------------------
            time_embedding = get_time_embedding(timesteps).to(cfg.device)  # (B,320)
            # Sinusoidal time embeddings are a way to turn the timestep t (a single integer) 
            # into a high‑dimensional vector using sine and cosine waves of different frequencies.
            """Computes sinusoidal time embeddings for each t.

            Shape (B, 320) in your UNet design"""

            # -------------------------
            # 5. Predict noise with diffusion UNet
            # -------------------------
            # diffusion(x_t, context, time_embedding) -> predicted noise
            model_pred = diffusion(noisy_latents, context, time_embedding)
            """Calls the diffusion UNet:

            Inputs:

            noisy_latents: x_t (B, 4, 64, 64)

            context: CLIP text embeddings (B, 77, Dim)

            time_embedding: (B, 320)

            Output: predicted noise ε̂."""

            # -------------------------
            # 6. Loss = MSE(predicted_noise, true_noise)
            # -------------------------
            loss = F.mse_loss(model_pred, noise)
            # Standard diffusion objective: MSE between predicted noise and true noise.

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # This prevents gradient accumulation between steps and slightly reduces memory use / can speed things up compared to zeroing the tensors
            # Reset gradients, backpropagate, and update UNet parameters.

            global_step += 1
            running_loss += loss.item()
            avg_loss = running_loss / (step + 1)

            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            """Track total steps, accumulate loss, compute average loss, and show it in tqdm."""

        # -------------------------
        # Save UNet checkpoint each epoch
        # -------------------------
        ckpt_path = os.path.join(cfg.output_dir, f"diffusion_flickr30k_epoch{epoch+1}.pt")
        torch.save(diffusion.state_dict(), ckpt_path)
        print(f"Saved UNet checkpoint to {ckpt_path}")
        """After each epoch:

        Build path like checkpoints/diffusion_flickr30k_epoch1.pt.

        Save only the diffusion UNet weights."""


if __name__ == "__main__":
    cfg = TrainConfig()
    train_unet_on_flickr30k(cfg)

