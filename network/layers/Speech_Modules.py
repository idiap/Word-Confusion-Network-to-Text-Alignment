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
import random

import torch
import torch.nn as nn

from network.layers.BERT_Modules import BERTNC
from network.layers.NN_Modules import Attention
from network.layers.pLSTM import CustomLSTM, CustomXMER, pLSTM


def MLM_target(tens, frac=0.10):  # tens -> [seq1,768; seq2,768; ...]
    mask_id = []
    target = torch.cat(tens, dim=0)
    offset = 0
    for i, ten in enumerate(tens):
        slen = ten.size(0)
        idx = random.sample(list(range(slen)), int(frac * slen))
        try:
            ten[idx] = 0.0
        except Exception as e:
            print(e)
        mask_id.extend([i + offset for i in idx])
        offset += slen
    return tens, target, sorted(mask_id)


class Listener(nn.Module):
    # This listener is the methos for encoding the audio
    def __init__(
        self, input_dim, pyr_layer, device, nlayer, nhead, enc_type="lstm", dropout=0.1
    ):
        super(Listener, self).__init__()
        # Speech feature extractor
        self.pBLSTM = pLSTM(
            input_dim, pyr_layer, "LSTM", device=device, dropout=dropout
        )
        self.bottle = nn.Linear((2**pyr_layer) * input_dim, 768)
        self.norm = nn.LayerNorm(768)
        # Text feature extractor
        self.tembed = BERTNC()
        self.tencoder = CustomLSTM(768, 1, "LSTM", device=device, dropout=dropout)
        self.tnorm = nn.LayerNorm(768)

        if enc_type == "lstm":
            self.encoder = CustomLSTM(
                768, nlayer, "LSTM", device=device, dropout=dropout
            )
            self.self_attn = Attention(768, nhead, dropout=dropout)
        else:
            self.encoder = CustomXMER(768, nlayer, device=device, dropout=dropout)
        self.enc_type = enc_type
        self.dropout = nn.Dropout(dropout)
        self.pyr_layer = pyr_layer

    def forward(self, input, is_train=False, mlm=False):
        # pyramid layers
        out_pyr, lens = self.pBLSTM(input)
        # --> 768
        in_enc_pad = self.norm(self.bottle(out_pyr))  # seq, bsz, 768
        # convert to list
        in_enc_list = [
            in_enc_pad.permute(1, 0, 2)[i][:l] for i, l in enumerate(list(lens))
        ]
        if mlm:
            in_enc_list, target, mask_id = MLM_target(in_enc_list)
        else:
            target, mask_id = None, None
        # remaining layers
        out_enc, mask = self.encoder(in_enc_list, is_train=is_train)
        if self.enc_type == "lstm":
            out_enc, _ = self.self_attn(out_enc, out_enc, mask.bool())
        return out_enc, mask, target, mask_id

    def forward_text(self, input, is_train=False, mlm=False):
        embed_out, mask1 = self.tembed(input)
        lens1 = (1 - mask1.long()).sum(dim=1).tolist()
        embed_list = [
            embed_out.permute(1, 0, 2)[i][:l] for i, l in enumerate(list(lens1))
        ]
        rnn_out, mask2 = self.tencoder(embed_list, is_train=False)
        rnn_out = self.tnorm(rnn_out)
        lens2 = (1 - mask2.long()).sum(dim=1).tolist()
        in_enc_list = [
            rnn_out.permute(1, 0, 2)[i][:l] for i, l in enumerate(list(lens2))
        ]

        if mlm:
            in_enc_list, target, mask_id = MLM_target(in_enc_list, frac=0.15)
        else:
            target, mask_id = None, None
        # remaining layers
        out_enc, mask = self.encoder(in_enc_list, is_train=is_train)
        if self.enc_type == "lstm":
            out_enc, _ = self.self_attn(out_enc, out_enc, mask.bool())
        return out_enc, mask, target, mask_id
