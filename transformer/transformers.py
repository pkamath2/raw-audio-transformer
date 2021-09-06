# MIT License

# Copyright (c) 2019 Peter Bloem

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn
import torch.nn.functional as F

from .modules import TransformerBlock

class GTransformer(nn.Module):
    """
    Transformer for generating audio.
    """
    def __init__(self, emb, heads, depth, seq_length, num_tokens, dropout=0.0):
        super().__init__()
    
        self.seq_length = seq_length
        self.emb = emb
        
        self.num_tokens = num_tokens
        
        self.token_embedding = nn.Linear(4, emb) 
        self.token_embedding_batchnorm = nn.BatchNorm1d(num_features=emb, track_running_stats=False) 
        self.token_embedding_activation = nn.ReLU()
        
        self.pos_embedding = nn.Linear(seq_length, emb)
        self.pos_embedding_batchnorm = nn.BatchNorm1d(num_features=emb, track_running_stats=False)
        self.pos_embedding_activation = nn.ReLU()
        
        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True, pos_embedding=self.pos_embedding, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_tokens)

    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        
        tokens = self.token_embedding(x) # Input to the model = batch_size X sample_length X 1 i.e. 16 X 512 X 1
        b, t, e = tokens.size() # Output from 'Word' embedding = batch_size X sample_length X embedding_size i.e. 16 X 512 X 128
        tokens = self.token_embedding_batchnorm(tokens.view(b, e, t)).view(b, t, e)
        tokens = self.token_embedding_activation(tokens)
        
        trange = torch.arange(t)
        trange = trange.cuda().float().view(1, -1)
        positions = self.pos_embedding(trange)[None, :, :].expand(b, t, e)# Output from 'Position' embedding = batch_size X sample_length X embedding_size i.e. 16 X 512 X 128
        positions = self.pos_embedding_batchnorm(positions.contiguous().view(b, e, t)).view(b, t, e)
        positions = self.pos_embedding_activation(positions)
        
        x = tokens + positions #+ conditioning # Output from 'Word' + 'Position' embedding = batch_size X sample_length X embedding_size i.e. 16 X 512 X 128 
        
        x = self.do(x)
        
        x = self.tblocks(x) # batch_size X sample_length X embedding_size i.e. 16 X 512 X 128

        x = self.toprobs(x.view(b*t, e)).view(b, t, self.num_tokens) # Output from Linear (in preparation of the softmax layer) = batch_size X sample_length X num_tokens i.e. 16 X 512 X 256

        x = F.log_softmax(x, dim=2) # Output from Log softmax = batch_size X sample_length X num_tokens i.e. 16 X 512 X 256

        return x