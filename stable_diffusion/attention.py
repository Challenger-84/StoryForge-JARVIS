import torch
from torch import nn
from torch.nn import functional as F 
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=False):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask = False):
        # x : (Batch_size, Seq_len, Dim)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interm_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        
        # (Batch_size, Seq_len, Dim) --> (Batch_size, Seq_len, Dim * 3) --> 3 tensors of shape (Batch_size, Seq_len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim = -1)

        # (Batch_size, Seq_len, Dim) --> (Batch_size, Seq_len, n_heads, d_head) --> (Batch_size, n_heads, Seq_len, d_head)
        q = q.view(interm_shape).transpose(1,2)
        k = k.view(interm_shape).transpose(1,2)
        v = v.view(interm_shape).transpose(1,2)

        # F.softmax((q @ k.tranpose(-1, -2) * (sequence_length ** -0.5))) @ v
        # (Batch_size, H, Seq_len, Dim/H) --> (Batch_size, H, Dim/H, Seq_len)
        weight = q @ k.tranpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim = -1)
        # (Batch_size, H, Seq_len, Seq_len) @ (Batch_size, H, Seq_len, d_head) --> (Batch_size, H, Seq_len, d_head)
        output = weight @ v
        # (Batch_size, H, Seq_len, d_head) --> (Batch_size, Seq_len, H, d_head) --> (Batch_size, Seq_len, Dim)
        output = output.transpose(1, 2).view(batch_size, sequence_length, d_embed)

        return self.out_proj(output)

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias = True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias = in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)

        self.out_proj = nn.Linear(d_embed, d_embed, bias= out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x : (latent) : (Batch_size, Seq_len_Q, Dim_Q)
        # y : (context) : (Batch_size, Seq_len_KV, Dim_KV) = (B, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interm_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interm_shape).transpose(1,2)
        k = k.view(interm_shape).transpose(1,2)
        v = v.view(interm_shape).transpose(1,2)

        weight = q @ k.tranpose(-1, -2)
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim = -1)
        # (Batch_size, H, Seq_len, Seq_len) @ (Batch_size, H, Seq_len, d_head) --> (Batch_size, H, Seq_len, d_head)
        output = weight @ v
        
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, d_embed)

        return self.out_proj(output)
