"""All-MLP and MolTriNet architectures, ported from the thesis source.

paper_liver reports All-MLP = 87,490 and MolTriNet = 875,122 trainable
parameters; those counts are the acceptance test for this reconstruction.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SMILES_CHARS = [
    'C', 'N', 'O', 'S', 'P', 'F', 'I', 'H',
    'c', 'n', 'o', 's', 'p', 'B', 'K', 'V', 'Y', 'W',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    '(', ')', '[', ']', '=', '#', '@', '+', '-', '/', '\\', '.', '%', '*',
    'Cl', 'Br', 'Si', 'Se', 'Na', 'Li', 'Ca', 'Mg', 'Fe', 'Zn', 'Cu',
]

CFG = dict(
    smiles_embed_dim=64, smiles_hidden_dim=128, smiles_output_dim=256,
    maccs_dim=167, maccs_output_dim=256, cnn_filters_1=32, cnn_filters_2=64,
    cnn_kernel_1=7, cnn_kernel_2=5, cam_reduction=4,
    physico_dim=9, physico_output_dim=64, transformer_heads=1, transformer_dim=32,
    fusion_hidden=256, dropout=0.4,
)


class SMILESTokenizer:
    def __init__(self, max_len=200):
        self.max_len = max_len
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for ch in SMILES_CHARS:
            if ch not in self.char_to_idx:
                self.char_to_idx[ch] = idx
                idx += 1
        self.vocab_size = len(self.char_to_idx)

    def encode(self, smiles):
        toks, i = [], 0
        while i < len(smiles):
            if i + 1 < len(smiles) and smiles[i:i + 2] in self.char_to_idx:
                toks.append(self.char_to_idx[smiles[i:i + 2]]); i += 2
            elif smiles[i] in self.char_to_idx:
                toks.append(self.char_to_idx[smiles[i]]); i += 1
            else:
                toks.append(self.char_to_idx['<UNK>']); i += 1
        toks = toks[:self.max_len]
        toks += [0] * (self.max_len - len(toks))
        return toks


class SimpleMLPModel(nn.Module):
    """All-MLP: 176 -> [256,128,64] with BatchNorm, dropout 0.4 -> 2."""

    def __init__(self, input_dim=176, hidden_dims=(256, 128, 64), num_classes=2, dropout=0.4):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SMILESEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, output_dim=256,
                 dropout=0.4, rnn_type='bilstm'):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        bidirectional = 'bi' in self.rnn_type
        rnn_class = nn.GRU if 'gru' in self.rnn_type else nn.LSTM
        self.rnn = rnn_class(embed_dim, hidden_dim, num_layers=2, batch_first=True,
                             bidirectional=bidirectional, dropout=dropout)
        mult = 2 if bidirectional else 1
        self.bn = nn.BatchNorm1d(hidden_dim * mult)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * mult, output_dim)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        if 'gru' in self.rnn_type:
            _, h = self.rnn(emb)
        else:
            _, (h, _) = self.rnn(emb)
        hidden = torch.cat([h[-2], h[-1]], dim=1) if 'bi' in self.rnn_type else h[-1]
        return self.fc(self.dropout(self.bn(hidden)))


class ChannelAttentionModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(nn.Linear(channels, channels // reduction), nn.ReLU(),
                                 nn.Linear(channels // reduction, channels))

    def forward(self, x):
        a = self.mlp(self.avg_pool(x).squeeze(-1))
        m = self.mlp(self.max_pool(x).squeeze(-1))
        return x * torch.sigmoid(a + m).unsqueeze(-1)


class MACCSEncoder(nn.Module):
    def __init__(self, input_dim=167, output_dim=256, filters_1=32, filters_2=64,
                 kernel_1=7, kernel_2=5, cam_reduction=4, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, filters_1, kernel_size=kernel_1, padding=kernel_1 // 2)
        self.bn1 = nn.BatchNorm1d(filters_1)
        self.conv2 = nn.Conv1d(filters_1, filters_2, kernel_size=kernel_2, padding=kernel_2 // 2)
        self.bn2 = nn.BatchNorm1d(filters_2)
        self.cam = ChannelAttentionModule(filters_2, reduction=cam_reduction)
        self.dropout = nn.Dropout(dropout)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(nn.Linear(filters_2 * 2, output_dim),
                                nn.BatchNorm1d(output_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))
        x = self.cam(x)
        avg = self.global_avg_pool(x).squeeze(-1)
        mx = self.global_max_pool(x).squeeze(-1)
        return self.fc(torch.cat([avg, mx], dim=1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PhysicoEncoder(nn.Module):
    def __init__(self, input_dim=9, output_dim=64, d_model=32, nhead=1, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_dim)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=d_model * 4,
                                           dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.bn = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(d_model, output_dim), nn.BatchNorm1d(output_dim),
                                nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.dropout(self.bn(x.mean(dim=1)))
        return self.fc(x)


class MolTriNet(nn.Module):
    """SMILES-BiLSTM + MACCS-CNN-CAM + physico-transformer fusion, 256-wide."""

    def __init__(self, vocab_size, num_classes=2, dropout=0.4):
        super().__init__()
        c = CFG
        self.smiles = SMILESEncoder(vocab_size, c['smiles_embed_dim'], c['smiles_hidden_dim'],
                                    c['smiles_output_dim'], dropout, 'bilstm')
        self.bn_smiles = nn.BatchNorm1d(c['smiles_output_dim'])
        self.maccs = MACCSEncoder(c['maccs_dim'], c['maccs_output_dim'], c['cnn_filters_1'],
                                  c['cnn_filters_2'], c['cnn_kernel_1'], c['cnn_kernel_2'],
                                  c['cam_reduction'], dropout)
        self.bn_maccs = nn.BatchNorm1d(c['maccs_output_dim'])
        self.physico = PhysicoEncoder(c['physico_dim'], c['physico_output_dim'],
                                      c['transformer_dim'], c['transformer_heads'], dropout)
        self.bn_physico = nn.BatchNorm1d(c['physico_output_dim'])
        total = c['smiles_output_dim'] + c['maccs_output_dim'] + c['physico_output_dim']
        self.fusion = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(total, c['fusion_hidden']),
            nn.BatchNorm1d(c['fusion_hidden']), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(c['fusion_hidden'], num_classes))

    def forward(self, smiles, maccs, physico):
        s = self.bn_smiles(self.smiles(smiles))
        m = self.bn_maccs(self.maccs(maccs))
        p = self.bn_physico(self.physico(physico))
        return self.fusion(torch.cat([s, m, p], dim=1))


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    tok = SMILESTokenizer()
    print(f"tokenizer vocab from fixed char list: {tok.vocab_size}")
    print(f"All-MLP params        : {n_params(SimpleMLPModel()):,}  (paper_liver: 87,490)")
    for v in (tok.vocab_size, 87):
        print(f"MolTriNet params vocab={v:<3d}: {n_params(MolTriNet(v)):,}  (paper_liver: 875,122)")
