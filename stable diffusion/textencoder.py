import torch.nn as nn
from transformers import BertTokenizer, BertModel

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation