import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import tiktoken
from dataclasses import dataclass

@dataclass
class TConfig:
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_embd: int = 512 # d_model or n_embd is size of embedding layer for text
    n_head: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1 # to prevent overfitting 

class TransformerModel(nn.Module):
    def __init__(self, Config):
        super().__init__()
        self.embedding = nn.Embedding(Config.vocab_size, Config.n_embd)
        self.transformer = nn.Transformer(Config.n_embd, Config.n_head, Config.num_encoder_layers, Config.num_decoder_layers, Config.dim_feedforward, Config.dropout)
        self.fc = nn.Linear(Config.n_embd, Config.vocab_size)
    
    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        transformer_out = self.transformer(src_emb, tgt_emb)
        output = self.fc(transformer_out)
        return output


model = TransformerModel(TConfig)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
loss = nn.CrossEntropyLoss()

