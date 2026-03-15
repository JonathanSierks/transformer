import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, vocab_size, num_heads, max_seq_length, embedding_dim = 300):
        super().__init__()

        # Define the three specific layers
        self.linear_K = nn.Linear(embedding_dim, embedding_dim)
        self.linear_Q = nn.Linear(embedding_dim, embedding_dim)
        self.linear_V = nn.Linear(embedding_dim, embedding_dim)

        self.linear_O = nn.Linear(embedding_dim, embedding_dim)
        
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim


    def forward(self, combined_embed):

        K = self.linear_K(combined_embed) # (B,T,emb)
        Q = self.linear_Q(combined_embed) # (B,T,emb)
        V = self.linear_V(combined_embed) # (B,T,emb)

        hemb = self.embedding_dim // self.num_heads

        K = torch.reshape(K, (K.size(0), K.size(1), self.num_heads, hemb)) # (B,T, k, hemb)
        Q = torch.reshape(Q, (Q.size(0), Q.size(1), self.num_heads, hemb)) # (B,T, k, hemb)
        V = torch.reshape(V, (V.size(0), V.size(1), self.num_heads, hemb)) # (B,T, k, hemb)

        K = K.permute(0,2,1,3) # (B, k, T, hemb)
        Q = Q.permute(0,2,1,3) # (B, k, T, hemb)
        V = V.permute(0,2,1,3) # (B, k, T, hemb)

        K_Q = Q @ K.transpose(-1, -2) # (B, k, T, T)

        K_Q_scaled = K_Q / math.sqrt(hemb) # (B, k, T, T)

        # causal mask so token t cannot attend to future tokens
        T = K_Q_scaled.size(-1)
        mask = torch.triu(
            torch.ones(T, T, device=combined_embed.device, dtype=torch.bool),
            diagonal=1
        ) # (T, T)

        K_Q_scaled = K_Q_scaled.masked_fill(mask, float("-inf")) # (B, k, T, T)

        W = F.softmax(K_Q_scaled, dim=-1) # (B, k, T, T)

        Z = W @ V # (B, k, T, hemb)

        Z = Z.permute(0,2,1,3) # (B, T, k, hemb)

        Z = torch.reshape(Z, (Z.size(0), Z.size(1), self.embedding_dim)) # (B, T, emb)

        O = self.linear_O(Z) # (B, T, emb)

        return O

class TransformerBlock(nn.Module):
    def __init__(self, vocab_size, num_heads, max_seq_length, embedding_dim = 300, dropout = 0.1):
        super().__init__()
        self.selfatt = MultiHeadSelfAttention(vocab_size, num_heads, max_seq_length, embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, 4*embedding_dim),
            nn.ReLU(),
            nn.Linear(4*embedding_dim, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, combined_embed):
        # Sub-layer 1: LN + SelfAtt + DropOut + Resid
        combined_embed_norm = self.ln1(combined_embed)  # (B, T, emb)
        att_out = self.selfatt(combined_embed_norm)     # (B, T, emb)
        att_out = self.dropout(att_out)                 # (B, T, emb)
        residual_added = att_out + combined_embed       # (B, T, emb)

        # Sub-layer 2: LN + FF + DropOut + Resid
        ln_2 = self.ln2(residual_added)                 # (B, T, emb)
        ff_out = self.ff(ln_2)                          # (B, T, emb)
        ff_out = self.dropout(ff_out)                   # (B, T, emb)
        out_resid2 = ff_out + residual_added            # (B, T, emb)

        return out_resid2

class AutoRegressiveTransformer(nn.Module):
    def __init__(self, vocab_size, num_heads, num_Tblocks, max_seq_length, embedding_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(vocab_size, num_heads, max_seq_length, embedding_dim, dropout)
            for _ in range(num_Tblocks)
        ])
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.Linear = nn.Linear(embedding_dim, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, x):
        #Trim sequences that are too long
        if x.size(1) > self.max_seq_length:
            x = x[:, -self.max_seq_length:]

        B, T = x.size()     # B = batch dimension, T = time dimension

        # Input and position embeddings
        input_embeddings = self.embedding(x) # (B, T, emb)
        time_dimension = torch.arange(T, device=x.device) # (T,)
        pos_embeddings = self.pos_embedding(time_dimension) # (T, emb)

        final_embeddings = input_embeddings + pos_embeddings # (B, T, emb) + (T, emb) -> broadcasting

        # Apply transformer blocks
        out = final_embeddings # (B, T, emb)
        for block in self.blocks:
            out = block(out) # (B, T, emb)

        # Linear Layer Classifier
        logits = self.Linear(out) # (B, T, vocab_size)
        return logits