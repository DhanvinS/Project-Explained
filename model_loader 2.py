from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter

"""This file is basically your “model loading” helper. It doesn’t run 
generation itself – it just builds the 4 main networks and loads their 
pre-trained weights so the rest of your code can use them."""

def preload_models_from_standard_weights(ckpt_path, device):
    # Defines a helper function to load all the models from a checkpoint

    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)
    # Calls the converter utility to load checkpoint weights and convert
    #  them into a state_dict compatible with PyTorch

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)
    """Instantiates the VAE encoder and moves it to the target device (GPU/CPU).
        Loads the pre-trained encoder weights.
        strict=True ensures that all expected layers in the model must match the 
        checkpoint exactly (no missing or extra weights)"""

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)


    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }

"""  Returns all 4 models in a dictionary, so the rest of the code can access them
 easily using keys 'clip', 'encoder', 'decoder', and 'diffusion'  """