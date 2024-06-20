#-------------------------! import statements !-------------------------
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import tiktoken
from dataclasses import dataclass
import time

#-------------------------! Hyperparameter !-------------------------
@dataclass
class TConfig:
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_embd: int = 512 # d_model or n_embd is size of embedding layer for text
    n_head: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1 # to prevent overfitting 


#-------------------------! transformer model !-------------------------
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

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f'using device {device}')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#-------------------------! Hyperparameter for dataloader !-------------------------
batch_size: int = 2
block_size: int = 512

#-------------------------! Data Loader Class !-------------------------
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        enc = tiktoken.get_encoding('gpt2')
        with open('input.txt', 'r') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, device = device)
        print(f'1 epoch = {len(self.tokens) // (B*T)} batches')

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B,T)

        self.current_position += B*T

        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0

        return x.to(device), y.to(device)

train_loader = DataLoaderLite(batch_size, block_size)

model = TransformerModel(TConfig)

# Improving performance using GPU
torch.set_float32_matmul_precision('high')
model.to(device)
model = torch.compile(model, backend='eager')

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
loss = nn.CrossEntropyLoss()

def train(model, train_loader, optimizer, loss, num_epochs=100):
    model.train()
    for i in range(num_epochs):
        t0 = time.time()
        src, tgt = train_loader.next_batch()
        optimizer.zero_grad()
        with torch.autocast(device_type = device, dtype = torch.bfloat16):
            output = model(src, tgt)

        loss = loss(output.view(-1, output.shape[-1]), tgt.view(-1))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000  # time difference is in milliseconds
        tokens_per_sec = (train_loader.B * train_loader.T)/ (t1 - t0)
        if (i + 1) % 10 == 0:
            print(f' step {i + 1}, loss: {loss.item()}, dt: {dt:.2f} ms, tok/sec {tokens_per_sec}')


train(model, train_loader, optimizer, loss, num_epochs=100)