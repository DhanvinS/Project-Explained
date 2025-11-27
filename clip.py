import torch
from torch import nn
from torch import functional as F
from attention import SelfAttention
# import statements

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int,n_token: int ):
        # takes 3 args 
        # n_vocab - no of unique tokens
        # n_embd - dimension of embedding vectors
        # n_token - no of tokens in each i/p (max tokens)
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        '''Creates a token embedding layer using nn.Embedding. This layer will
          learn a unique vector of length n_embd for each token ID in the input,
            with n_vocab possible tokens.'''
        
        self.position_embedding = nn.Parameter(torch.zeros(n_token, n_embd))
        '''Creates a positional embedding matrix as a learnable parameter with shape (n_token, n_embd). 
        Each position in the sequence has a learnable vector of the same dimensionality as the token 
        embeddings.
        torch.zeros((n_token, n_embd)) initializes this matrix with zeros, but since it's wrapped in
          nn.Parameter, its values will be updated during training.'''
        
    def forward(self, tokens):
        x = self.token_embedding(tokens)
        # assigns the token embeddings of i/p to x
        x += self.position_embedding
        # adds the positional embeddings
        return x
    


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        # n_heads - no of attention heads for self attention
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        # Layer normalize input features before self attention step
        self.attention = SelfAttention(n_head, n_embd)
        #import self attention we created earlier

        self.layernorm_2 = nn.LayerNorm(n_embd)
        # does same as first norm1 but for feed forward network
        self.linear_1 = nn.Linear(n_embd, 4* n_embd)
        #expands dimension from n_embd to 4 * n_embd
        '''Gives the model a "richer," higher-dimensional space to perform complex, 
        non-linear transformations (like a non-linear projection). This allows the feed-forward layer 
        to model more complex relationships that one small linear layer couldn't.'''
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)
        # brings back the dimension Linaer method intros linearity
        '''Self-attention softmax → makes sure attention weights are valid probabilities and sums to 1.
           LayerNorm → makes sure the token embeddings themselves have stable distributions before 
           entering attention or feedforward.'''

    def forward(self, x):
        residue = x
        """Save the input tensor (x) into a variable called residue. This is for a residual 
        connection: after processing, you’ll add the original input back to the output to help 
        with gradient flow and training stability."""
        ### SELF ATTENTION LAYER ###
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        # in the init constructor we were just calling the layer here wer actually normalizing it 
        x += residue
        '''You add the original input back after the block (after normalization and self-attention)
          to ensure the network has a direct path for information and gradients—leading to better, 
          deeper, and more robust models.'''


        ### FEED FORWARD LAYER ###
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation
        """Apply an activation function (QuickGELU) for non-linearity,
          making the model more powerful and able to learn complex things."""
        x = self.linear_2(x)
        x += residue
        return x
    

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        # vocab size - 49408 - no of unique tokensthe model can recognise
        # dimension(embedding) size of 768 
        # max no of tokens(sequence length) 77 - max lenght of token that a model can 
        # process in a single i/p sample
        self.layers = nn.ModuleList([CLIPLayer(12, 768)] for i in range(12))
        # stack 12 transformer layers 
        # 12 attention heads per block(layer) and then there are 12 blocks(layers) of these 
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        # ensure token are of type torch.lang if not covnert
        state = self.embedding(tokens)
        # pass token id thru embd layer and add both token and pos embd
        # reulting tensor - (batch_size, sequence_length, embedding_dim)
        for layer in self.layers: 
            state = layer(state)
        # loop thru each trans layer 
        # each layer adds its understanding of i/p
        output = self.layernorm(state)
        # apply final layer normalization produce final token embeddings and return
        return output
    






    


