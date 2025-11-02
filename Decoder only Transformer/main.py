import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import lightning as L

# Toy vocabulary
tkn_2_id = {
     'soft': 0, 'kitty': 1, 'warm': 2, 'little': 3,
     'ball': 4, 'of': 5, 'fur': 6, 'happy': 7,
     'sleepy': 8, 'pur': 9, '<EOS>': 10
 }
id_2_tkn = {i: t for t, i in tkn_2_id.items()}

# Sample dataset (B x T)
inputs = torch.tensor([
     [tkn_2_id['soft'],
      tkn_2_id['kitty'], 
      tkn_2_id['warm'], 
      tkn_2_id['kitty'], 
      tkn_2_id['<EOS>'], 
      tkn_2_id['little'], 
      tkn_2_id['ball'], 
      tkn_2_id['of'], 
      tkn_2_id['fur'], 
      tkn_2_id['<EOS>']],
     [tkn_2_id['happy'], 
      tkn_2_id['kitty'], 
      tkn_2_id['sleepy'], 
      tkn_2_id['kitty'], 
      tkn_2_id['<EOS>'], 
      tkn_2_id['pur'], 
      tkn_2_id['pur'], 
      tkn_2_id['pur'], 
      tkn_2_id['<EOS>'], 
      tkn_2_id['<EOS>']]
 ])
labels = torch.tensor([
     [tkn_2_id['kitty'], 
      tkn_2_id['warm'], 
      tkn_2_id['kitty'], 
      tkn_2_id['<EOS>'], 
      tkn_2_id['little'], 
      tkn_2_id['ball'], 
      tkn_2_id['of'], 
      tkn_2_id['fur'], 
      tkn_2_id['<EOS>'], 
      tkn_2_id['<EOS>']],
     [tkn_2_id['kitty'], 
      tkn_2_id['sleepy'], 
      tkn_2_id['kitty'], 
      tkn_2_id['<EOS>'], 
      tkn_2_id['pur'], 
      tkn_2_id['pur'], 
      tkn_2_id['pur'], 
      tkn_2_id['<EOS>'], 
      tkn_2_id['<EOS>'], 
      tkn_2_id['<EOS>']]
 ])

dataset = TensorDataset(inputs, labels)
loader = DataLoader(dataset, batch_size=2)

# Positional Encoding
class PE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

# Attention
class Attention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

# Decoder-only Transformer
class DecoderOnlyTransformer(L.LightningModule):
    def __init__(self, vocab_size: int, d_model: int = 16, max_len: int = 10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PE(d_model, max_len)
        self.attn = Attention(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        B, T = x.size()
        emb = self.embedding(x)           # [B, T, D]
        emb = self.pe(emb)               # [B, T, D]
        # causal mask (lower triangular)
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn_out = self.attn(emb, mask=mask)    # [B, T, D]
        out = emb + attn_out                    # residual
        logits = self.fc(out)                   # [B, T, V]
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch                             # [B, T], [B, T]
        logits = self(x)                         # [B, T, V]
        # reshape for loss: [B*T, V] vs. [B*T]
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

# Instantiate and train
model = DecoderOnlyTransformer(vocab_size=len(tkn_2_id), d_model=16, max_len=inputs.size(1))
trainer = L.Trainer(max_epochs=30)
trainer.fit(model, loader)

# autoregressive generation
def generate(model, start_tokens: list, max_len: int = 10):
    model.eval()
    ids = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0)  # [1, T]
    with torch.no_grad():
        for _ in range(max_len - ids.size(1)):
            logits = model(ids)                # [1, T, V]
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            if next_id.item() == tkn_2_id['<EOS>']:
                break
            ids = torch.cat([ids, next_id], dim=1)
    return ids.squeeze(0).tolist()

#ex
output_ids = generate(model, [tkn_2_id['soft'], tkn_2_id['kitty'], tkn_2_id['warm'], tkn_2_id['<EOS>']], max_len=10)
print("Generated:", [id_2_tkn[i] for i in output_ids])





