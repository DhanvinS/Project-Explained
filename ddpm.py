import torch
import numpy as np

class DDPMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        # initialize sampler with diffusion parameters
        # In this context, the generator is the random number generator object that creates the initial noise for the diffusion process and controls reproducibility via its seed.
        # We have start beta and end beta and we divide up the range into diff beta values over the 1000 steps
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        # This line is creating a 1D tensor of betas that increase quadratically over time, using a trick with square roots to make the square roots of betas increase linearly
        self.alphas = 1.0 - self.betas
        # initializing alpha and beta values
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        #ᾱ_t = product_{i=1}^t α_i-> cumulative product, used in formulas for forward q(x_t|x₀).
        #This tells the total signal retention after t steps
        # we can sample any point in forward process in one shot using this
        self.one = torch.tensor(1.0)
        #That line just creates a constant tensor with the scalar value 1.0 that the schedule code can reuse
        self.generator = generator
        '''Stores the torch.Generator you passed in so the scheduler/sampler can use it later to draw random noise in a reproducible way (same seed → same samples)'''
        self.num_train_timesteps = num_training_steps
        # Saves how many diffusion timesteps the model was trained with (e.g., 1000); this is the length of your beta/alpha schedule and defines the valid range of timestep indices
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
        """The last line builds a tensor that lists all the diffusion timesteps in reverse order, so the sampler knows which step index to use at each denoising step"""

    def set_inference_timesteps(self, num_inference_steps=50):
        """This function picks which timesteps to actually use at inference when you want fewer steps than the model was trained with."""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # Computes how many training steps you “skip” between each inference step (e.g., 1000 train steps → 50 inference steps ⇒ step_ratio = 20)
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        """Make a NumPy array of which training steps to use at inference: evenly spaced indices (by step_ratio), then reverse it so you go from very noisy to clean (big index → 0), and ensure it’s an int64 array"""
        self.timesteps = torch.from_numpy(timesteps)
        #Convert that NumPy array into a PyTorch tensor and store it, so the sampler will loop over exactly those timesteps during denoising

        """Y we keep same consistent datatype all over our models 
           In simple terms: if all numbers (tensors) are the same kind (for example, 32‑bit floats),
             PyTorch can do the math correctly, avoid errors from mixing types, and use the GPU efficiently"""

    def _get_previous_timestep(self, timestep: int) -> int:
        # Defines a function that takes the current timestep index (an integer) and returns another integer: the previous timestep it should jump to
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        """Computes how far to step back: it subtracts step_ratio = num_train_timesteps // num_inference_steps from the current timestep. So if training had 1000 steps and inference uses 50, step_ratio = 20, and this moves from, say, 980 → 960 → 940, etc"""
        return prev_t
        
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        # this func computes how much random noise to add at this step when going from xt to xt-1
        prev_t = self._get_previous_timestep(timestep)
        # finds which earlier timestep this should go
        alpha_prod_t = self.alphas_cumprod[timestep]
        # use the cumprod func to get how much of the image is left at this timestamp t
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        # This is alpha t-1 for the previous step; if there is no previous step (t is the first), it uses 1.0 as a safe default
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        # This recovers the “per-step” noise amount between t and prev_t (a kind of effective beta T )from the cumulative products
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        # applies the ddpm formula to get variance of the gaussian noise to add when sampling
        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)
        # Makes sure the variance is never exactly zero or negative, to avoid numerical issues
        # variance here refers to variance of noise
        return variance
    
    def set_strength(self, strength=1):
        # for image to image 
        """
            Set how much noise to add to the input image. 
            More noise (strength ~ 1) means that the output will be further from the input image.
            Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        # start_step is the number of noise levels to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
        # saves start step
        # helps us skip time steps to directly go to any timestep we desire

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        """timestep: current step t
            latents: z_t (noisy latent at step t)
            model_output: predicted noise ε̂ from U-Net """
        t = timestep
        # current step
        prev_t = self._get_previous_timestep(t)
        # prev step
        alpha_prod_t = self.alphas_cumprod[t]
        # product of alphas function
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        # alpha for prev step
        beta_prod_t = 1 - alpha_prod_t
        # beta formula at t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        # prev timestep beta
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        # ratio of current t alpha to t-1 alpha gives the alpha changed
        current_beta_t = 1 - current_alpha_t
        # remember all the formulas we initialized uses all of them here

       
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        """Formula: x₀̂ = (x_t - √(1-ᾱ_t) ε̂) / √ᾱ_t
           Uses predicted noise to estimate clean latent at t=0."""
        # estimate clean image using formula
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        """pred_original_sample_coeff = how much of the clean estimate x0 to use.current_sample_coeff = how much of the current noisy sample xt to keep."""
        
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
        # our best guess of next less noisy step

     
        variance = 0
        """this below snippet Adds random noise scaled by predicted variance
            Ensures diversity in generated images"""
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            """Think of DDPM as doing two things at once in the backward process:
                    Denoising (mean part) – move xt toward a cleaner sample using the model’s predicted noise.
                    Sampling (random part) – add back a controlled amount of noise so that you are drawing from a probability distribution, not following a single fixed path"""
            variance = (self._get_variance(t) ** 0.5) * noise
            # re adding noise here we add at every backward step
            # So: the model uses its prediction to reduce noise on average, but still injects a small, scheduled amount of randomness so the reverse process stays a stochastic generative model, not just a fixed deblurring algorithm
            # Yes, the amount of noise is chosen so that each step exactly follows a Gaussian distribution with a specific variance that was fixed when designing the scheduler
        pred_prev_sample = pred_prev_sample + variance
        # add variance
        return pred_prev_sample
    
    def add_noise(self,original_samples: torch.FloatTensor, timesteps: torch.IntTensor,) -> torch.FloatTensor:
        # forward diffusion
        # does not take part in backward
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        # the while statements do this Those lines just reshape things so “one number per batch item” can be multiplied with a whole image tensor

        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        # this is epsilon noise = epsilon from our formulas 
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

        

    
