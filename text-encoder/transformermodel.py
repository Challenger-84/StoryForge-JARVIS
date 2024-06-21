#-------------------------! import statements !-------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BartTokenizer, BartForConditionalGeneration
import time
from dataclasses import dataclass

#-------------------------! Hyperparameter !-------------------------
@dataclass
class BartConfig:
    model_name: str = "facebook/bart-base"
    batch_size: int = 2
    block_size: int = 1024 

#-------------------------! Load BART Model and Tokenizer !-------------------------
config = BartConfig()
tokenizer = BartTokenizer.from_pretrained(config.model_name)
model = BartForConditionalGeneration.from_pretrained(config.model_name)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f'using device {device}')

model.to(device)
model = torch.compile(model, backend='eager')

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

#-------------------------! Data Loader Class !-------------------------
class DataLoaderLite:
    def __init__(self, B, T, tokenizer):
        self.B = B
        self.T = T
        self.tokenizer = tokenizer

        with open('input.txt', 'r') as f:
            text = f.read()
        tokens = tokenizer(text, return_tensors="pt", max_length=T, truncation=True, padding="max_length")["input_ids"]
        self.tokens = tokens.to(device)

        self.num_batches = max(len(self.tokens) // B, 1)
        print(f'1 epoch = {self.num_batches} batches')

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        start_idx = self.current_position
        end_idx = start_idx + B

        x = self.tokens[start_idx:end_idx]
        y = self.tokens[start_idx:end_idx]

        self.current_position += B

        if self.current_position + B > len(self.tokens):
            self.current_position = 0

        return x, y

train_loader = DataLoaderLite(config.batch_size, config.block_size, tokenizer)

def train(model, train_loader, optimizer, num_epochs=100):
    model.train()
    for i in range(num_epochs):
        t0 = time.time()
        src, tgt = train_loader.next_batch()
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            output = model(input_ids=src, labels=tgt)
            loss = output.loss

        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000  # time difference is in milliseconds
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        
        print(f'step {i + 1}, loss: {loss.item()}, dt: {dt:.2f} ms, tok/sec {tokens_per_sec}')

train(model, train_loader, optimizer, num_epochs=50)


def save_model(model, path):
  model.save_pretrained(path)
  tokenizer.save_pretrained(path)
  print(f'Model saved to {path}')

path = "./text_model"  
save_model(model, path)
