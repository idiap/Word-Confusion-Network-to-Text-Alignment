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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.augment import SpecAugment
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def extract2(tens, out_lens):
    out = []
    for i, ten in enumerate(tens):
        out.append(ten[: out_lens[i]])
    return out


def spec(input_x):
    aug = SpecAugment(
        time_warp=False, freq_mask_width=(0, 100), time_mask_width=(0, 10)
    )
    pack = pack_sequence(input_x, enforce_sorted=False)
    out_pad, lens = pad_packed_sequence(pack, batch_first=True)
    out_pad_ = aug(out_pad)
    return extract2(out_pad_, lens)


def sni(input_x, low=0.0, high=0.4):
    pack = pack_sequence(input_x, enforce_sorted=False)
    out_pad, lens = pad_packed_sequence(pack, batch_first=True)
    M = out_pad.size(0)
    idx = list(range(M))
    random.shuffle(idx)
    noise = out_pad[idx]
    lens_ = [max(lens[i], lens[j]) for i, j in enumerate(idx)]
    lam = (
        torch.from_numpy(np.random.uniform(low, high, (M, 1, 1)))
        .float()
        .to(out_pad.get_device())
    )
    out_pad_ = (1.0 - lam) * out_pad + lam * noise
    return extract2(out_pad_, lens_)


def mix(input_x, labels, dist, low=0.5, high=15.0):
    pack = pack_sequence(input_x, enforce_sorted=False)
    out_pad, lens = pad_packed_sequence(pack, batch_first=True)
    M = out_pad.size(0)
    idx = list(range(M))
    random.shuffle(idx)
    noise = out_pad[idx]
    labels_ = labels[idx]
    lens_ = [max(lens[i], lens[j]) for i, j in enumerate(idx)]
    alpha = []
    for lr in dist:
        if lr:
            alpha.append(high)
        else:
            alpha.append(low)
    lam = np.random.beta(alpha, alpha)
    lam = (
        torch.from_numpy(np.maximum(lam, 1.0 - lam))
        .float()
        .unsqueeze(1)
        .unsqueeze(2)
        .to(out_pad.get_device())
    )
    out_pad_ = lam * out_pad + (1.0 - lam) * noise
    labels_mix = lam.squeeze(1) * labels + (1.0 - lam.squeeze(1)) * labels_
    return extract2(out_pad_, lens_), labels_mix


def get_mask(lens):
    mask = torch.ones(len(lens), max(lens))
    for i, l in enumerate(lens):
        mask[i][:l] = 0.0
    return mask


class pLSTMLayer(nn.Module):
    # LSTM layer for pLSTM
    # Step 1. Reduce time resolution to half
    # Step 2. Run through pLSTM
    # Note the input should have timestep%2 == 0
    def __init__(self, input_dim, hidden_dim, rnn_unit="LSTM", dropout=0.1):
        super(pLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn, rnn_unit.upper())

        # feature dimension will be doubled since time resolution reduction
        self.pLSTM = self.rnn_unit(
            input_dim, hidden_dim, 1, bidirectional=True, batch_first=False
        )

        self.linear1 = nn.Linear(2 * hidden_dim, 4 * hidden_dim)
        self.linear2 = nn.Linear(4 * hidden_dim, 2 * hidden_dim)

        self.norm1 = nn.LayerNorm(2 * hidden_dim)
        self.norm2 = nn.LayerNorm(2 * hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        red_x = []
        # Reduce time resolution
        for x in input:
            timestep = x.size(0)
            feature_dim = x.size(1)
            red_x.append(x.contiguous().view(int(timestep / 2), feature_dim * 2))
        # forward lstm
        pack = pack_sequence(red_x, enforce_sorted=False)
        input_padded, _ = pad_packed_sequence(pack)
        output, hidden = self.pLSTM(pack)
        output_padded, lens_unpacked = pad_packed_sequence(output)
        # Add and norm
        out = self.norm1(input_padded + self.dropout(output_padded))
        # MLP
        out2 = self.linear2(self.dropout(F.relu(self.linear1(self.dropout(out)))))
        # Add and norm
        out = self.norm2(out + self.dropout(out2))
        # post-process
        output_padded_batchf = self.dropout(out).permute(1, 0, 2)
        out_list = [
            output_padded_batchf[i][:l] for i, l in enumerate(list(lens_unpacked))
        ]
        return out_list, hidden


class pLSTM(nn.Module):
    # Listener is a pLSTM stacking n layers to reduce time resolution 2^n times
    def __init__(self, input_dim, nlayer, rnn_unit, device=None, dropout=0.1):
        super(pLSTM, self).__init__()
        # Listener RNN layer
        self.nlayer = nlayer
        assert self.nlayer >= 1, "Listener should have at least 1 layer"

        self.pLSTM_layer0 = pLSTMLayer(
            2 * input_dim, input_dim, rnn_unit=rnn_unit, dropout=dropout
        )
        input_dim = 2 * input_dim
        for i in range(1, self.nlayer):
            setattr(
                self,
                "pLSTM_layer" + str(i),
                pLSTMLayer(
                    2 * input_dim, input_dim, rnn_unit=rnn_unit, dropout=dropout
                ),
            )
            input_dim = 2 * input_dim

    def forward(self, input):
        output, _ = self.pLSTM_layer0(input)
        for i in range(1, self.nlayer):
            output, _ = getattr(self, "pLSTM_layer" + str(i))(output)
        packed = pack_sequence(output, enforce_sorted=False)
        output_padded, lens = pad_packed_sequence(packed)
        return output_padded, lens


class CustomLSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_unit="LSTM", dropout=0.1):
        super(CustomLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn, rnn_unit.upper())

        self.pLSTM = self.rnn_unit(
            input_dim, hidden_dim, 1, bidirectional=True, batch_first=False
        )

        self.linear1 = nn.Linear(2 * hidden_dim, 4 * hidden_dim)
        self.linear2 = nn.Linear(4 * hidden_dim, 2 * hidden_dim)

        self.norm1 = nn.LayerNorm(2 * hidden_dim)
        self.norm2 = nn.LayerNorm(2 * hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # forward lstm
        pack = pack_sequence(input, enforce_sorted=False)
        input_padded, _ = pad_packed_sequence(pack)
        output, hidden = self.pLSTM(pack)
        output_padded, lens_unpacked = pad_packed_sequence(output)
        # Add and norm
        out = self.norm1(input_padded + self.dropout(output_padded))
        # MLP
        out2 = self.linear2(self.dropout(F.relu(self.linear1(self.dropout(out)))))
        # Add and norm
        out = self.norm2(out + self.dropout(out2))
        # post-process
        output_padded_batchf = self.dropout(out).permute(1, 0, 2)
        out_list = [
            output_padded_batchf[i][:l] for i, l in enumerate(list(lens_unpacked))
        ]
        return out_list, hidden


class CustomLSTM(nn.Module):
    def __init__(self, input_dim, nlayer, rnn_unit, device=None, dropout=0.1):
        super(CustomLSTM, self).__init__()
        self.nlayer = nlayer
        self.LSTM_layer0 = CustomLSTMLayer(
            input_dim, int(input_dim / 2), rnn_unit=rnn_unit, dropout=dropout
        )
        for i in range(1, self.nlayer):
            setattr(
                self,
                "LSTM_layer" + str(i),
                CustomLSTMLayer(
                    input_dim, int(input_dim / 2), rnn_unit=rnn_unit, dropout=dropout
                ),
            )
        self.device = device

    def forward(self, input, is_train=False):
        mix_layer = random.sample([0, 1, 2, 3, 4, 5], 1)
        if 0 in mix_layer and is_train:
            input = spec(input)
        output, _ = self.LSTM_layer0(input)
        for i in range(1, self.nlayer):
            if i in mix_layer and is_train:
                output = spec(output)
            output, _ = getattr(self, "LSTM_layer" + str(i))(output)
        packed = pack_sequence(output, enforce_sorted=False)
        output_padded, lens = pad_packed_sequence(packed)
        mask = get_mask(list(lens))
        return output_padded, mask.to(self.device)


class CustomXMERLayer(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super(CustomXMERLayer, self).__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=12, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(1000, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, input, position=False):
        # pad
        pack = pack_sequence(input, enforce_sorted=False)
        input_padded, lens = pad_packed_sequence(pack)
        mask = get_mask(list(lens)).to(input_padded.get_device())
        # forward xmer
        if position:
            ps = torch.arange(input_padded.size(0)).to(input_padded.get_device())
            ps_embed = self.embed(ps).unsqueeze(1)
            input_padded = self.norm(input_padded + ps_embed)
        output = self.layer(input_padded, src_key_padding_mask=mask.bool())
        # unpad
        output_padded_batchf = self.dropout(output).permute(1, 0, 2)
        out_list = [output_padded_batchf[i][:l] for i, l in enumerate(list(lens))]
        return out_list


class CustomXMER(nn.Module):
    def __init__(self, input_dim, nlayer, device=None, dropout=0.1):
        super(CustomXMER, self).__init__()
        self.nlayer = nlayer
        self.XMER_layer0 = CustomXMERLayer(input_dim, dropout=dropout)
        for i in range(1, self.nlayer):
            setattr(
                self, "XMER_layer" + str(i), CustomXMERLayer(input_dim, dropout=dropout)
            )
        self.device = device

    def forward(self, input, is_train=False):
        mix_layer = random.sample([0, 1, 2, 3, 4, 5], 1)
        if 0 in mix_layer and is_train:
            input = spec(input)
        output = self.XMER_layer0(input, position=True)
        for i in range(1, self.nlayer):
            if i in mix_layer and is_train:
                output = spec(output)
            output = getattr(self, "XMER_layer" + str(i))(output)
        packed = pack_sequence(output, enforce_sorted=False)
        output_padded, lens = pad_packed_sequence(packed)
        mask = get_mask(list(lens))
        return output_padded, mask.to(self.device)
