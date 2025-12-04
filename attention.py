import torch
from torch import nn
from torch.nn import functional as F
import math

"""These two import statements are for using PyTorch's neural network toolkit:

from torch import nn
This imports the torch.nn module and lets you access it as nn. The module provides all the building blocks for neural networks in PyTorch, such as:

Layers (e.g., nn.Linear, nn.Conv2d)
Containers (e.g., nn.Sequential)
Predefined loss functions (e.g., nn.CrossEntropyLoss)
The base class for models (nn.Module)
from torch.nn import functional as F
This imports the functional API and aliases it as F. The F namespace provides stateless functions for operations like:

Activation functions (e.g., F.relu, F.softmax, F.sigmoid)
Other layer operations that don’t need to keep parameters or state."""

class SelfAttention(nn.Module):
  def ___init___(self, n_heads, d_embed, in_proj_bias = True, out_proj_bias = True):
    """n_heads = no of attention heads, d_head = embedding dimension
       n_proj_bias, out_proj_bias - indicates whether i/p o/p peojectiosn must have bias """
    super().__init__()
    #calls constructor of parent class -
    '''The parent initialization (e.g., from nn.Module) contains all the essential setup
     for parameter tracking, submodule registration, device management, and other features
      so your custom layer or model integrates seamlessly with PyTorch’s functionality.'''
    self.in_proj = nn.Linear(d_embed, 3*d_embed, bias = in_proj_bias)
    #projs input into one (Q, K ,V) for ease can be split later for model use
    self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
    # we proj i/p into one no we put it back while giving as o/p
    self.n_heads = n_heads
    self.d_embed = d_embed // n_heads
    # Slicing hte dimensions per head
    # Computes the dimension of each attention head by dividing the total embedding size by the number of heads.

  def forward(self, x, causal_mask = False):
    # causal mask - future blocking mask, x is i/p tensor
    input_shape = x.shape
    batch_size, sequence_length, d_embed = input_shape
    #wer gonna store the original input shape in a var so that we can later use it to get back o/p
    #unpack into 3 (B S D)for easy reference originally this was combiens into 1
    interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
    # reshape into 4 (B S D1 D2)
    q, k, v = self.in_proj(x).chunk(3, dim = -1)
    #projects x into queries, key and values
    # value is the original word embeddings
    # we later gonna multiply softmaxed vector to v to get new embeddings for the word
    q = q.view(interim_shape).transpose(1, 2)
    k = k.view(interim_shape).transpose(1, 2)
    v = v.view(interim_shape).transpose(1, 2)
    # change shape from bsd to interrim
    weight = q @ k.transpose(-1, -2)
    # @ in pytorch is matrix multiplication
    # calculate attention scores Q * K

    if causal_mask:
      mask = torch.ones_like(weight, dtype = torch.bool).triu(1)
      weight.masked_fill_(mask, -torch.inf)
      # apply causal mask ie make future i/p - infinity to not affect curr i/p
      # y minus infinity so that added softmax probability comes to 1
      # softmax of - inf becoems 0 future tokens gets no weights
    weight /= math.sqrt(self.d_head)
    # divide bt root of dimensionality stabilizes gradient and softmax
    # scales down attentionto prevent gradients to becoem too small or usntable
    # if we dont do this probabilities will go near 0 or 1 wont spread
    weight = F.softmax(weight, dim = -1)
    #softmax it
    output = weight @ v
    #multiply to get new embedding
    output = output.transpose(1, 2)
    # transpose axes
    output = output.reshape(input_shape)
    # get o/p into original i/p shape
    output = self.out_proj(output)
    return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(self.d_head)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()

        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output
     