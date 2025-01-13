#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2024> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Esau Villatoro-Tello <esau.villatoro@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
"""
Some sections of this code were inspired by the paper from Vishal Sunder, Eric Fosler-Lussier, Samuel Thomas, Jeff Kuo, Brian Kingsbury, "Tokenwise Contrastive Pretraining for Finer Speech-to-BERT Alignment in End-to-End Speech-to-Intent Systems", INTERSPEECH 2022
"""

import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, input_dim, nhead, dim_feedforward=2048, dropout=0.1):
        super(Attention, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, mask):
        src, attn = self.self_attn(Q, K, K, key_padding_mask=mask)
        # return _rest_of_forward(Q, self.norm1, self.linear1, self.linear2, self.norm2)
        # Add and norm
        src = Q + self.dropout(src)
        src = self.norm1(src)
        # MLP
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        # Add and norm
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src, attn


class MLP(nn.Module):
    def __init__(self, pyr_layer, input_dim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(768, 768)
        self.l2 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        out1 = F.relu(self.l1(self.dropout(input)))
        out2 = self.l2(self.dropout(out1))
        return out2
