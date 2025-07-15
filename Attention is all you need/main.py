import torch
import torch.nn as nn 
import torch.nn.functional as f
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import lightning as L


tkn_2_id = { 'soft':0,  'kitty':1, 'warm':2, 'kitty':3 ,'little':4, 'ball':5, 'of':6, 'fur':7, 'happy':8, 'kitty':9, 'sleepy':10, 'kitty':11, 'pur':12, 'pur':13, 'pur':14, '<EOS>':15}
id_2_tkn = dict(map(reversed, tkn_2_id.items()))
inputs= torch.tensor([[tkn_2_id["soft"], ##ip1: softkitty warm kitty <EOS> little ball of fur
                      tkn_2_id["kitty"],
                      tkn_2_id["little"],
                      tkn_2_id["warm"],
                      tkn_2_id["kitty"],
                      tkn_2_id["<EOS>"],
                      tkn_2_id["little"],
                      tkn_2_id["ball"],
                      tkn_2_id["of"],
                      tkn_2_id["fur"]],

                     [tkn_2_id["happy"],#ip2: happy kitty sleepy kitty<EOS> pur pur pur
                      tkn_2_id["kitty"],
                      tkn_2_id["sleepy"],
                      tkn_2_id["kitty"],
                      tkn_2_id["<EOS>"],
                      tkn_2_id["pur"],
                      tkn_2_id["pur"],
                      tkn_2_id["pur"]]])


labels= torch.tensor([[[tkn_2_id["kitty"],
                      tkn_2_id["little"],
                      tkn_2_id["warm"],
                      tkn_2_id["kitty"],
                      tkn_2_id["<EOS>"],
                      tkn_2_id["little"],
                      tkn_2_id["ball"],
                      tkn_2_id["of"],
                      tkn_2_id["fur"],
                      tkn_2_id["<EOS>"]],

                     [tkn_2_id["kitty"],
                      tkn_2_id["sleepy"],
                      tkn_2_id["kitty"],
                      tkn_2_id["<EOS>"],
                      tkn_2_id["pur"],
                      tkn_2_id["pur"],
                      tkn_2_id["pur"],
                      tkn_2_id["<EOS>"]]]])
dataset = TensorDataset(inputs, labels)
dataloader= DataLoader(dataset)

#positional encoding
class PE(nn.Module):
    def __init__(self, d_model=2, max_len=6):
        super().__init__()
        pe= torch.zeros(max_len, d_model)
        position = torch.arange(start =0, end = max_len, step=1).float().unsqueeze(1)
        embedding_idx= torch.arange(start=0, end = d_model, step=2).float()
        div_term = 1/ torch.tensor(10000)**(embedding_idx /d_model)

        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position* div_term)

        self.register_buffer('pe', pe)
    def forward(self, word_embeddings):
        return word_embeddings+self.pe[:word_embeddings.size(0),:] 
    
#Attention
class Attention(nn.Module):
    def __init__(self, d_model=2):
        super.__init__()
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_k= nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.r_dim = 0
        self.c_dim = 1

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        q= self.w_q(encodings_for_q)
        k= self.w_k(encodings_for_k)
        v= self.w_k(encodings_for_v)

        sims= torch.matmul(q,k.transpose(dim0= self.r_dim, dim1= self.c_dim))
        scaled_sims = sims/torch.tensor(k.size(self.c_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
        attention_percents = f.softmax(scaled_sims, dim=self.c_dim)
        attention_scores = torch.matmul(attention_percents, v)
        return attention_scores

