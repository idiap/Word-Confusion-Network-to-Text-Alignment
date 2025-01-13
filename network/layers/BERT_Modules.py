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
from transformers import BertForMaskedLM


class BERTNC(nn.Module):
    def __init__(self):
        super(BERTNC, self).__init__()
        # We only keep the BertEmbedding network, which takes care of adding positional embeddings and generates
        # some non-contextual embeddings
        # BertEmbeddings(
        #    (word_embeddings): Embedding(30522, 768, padding_idx=0)
        #    (position_embeddings): Embedding(512, 768)
        #    (token_type_embeddings): Embedding(2, 768)
        #    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        #    (dropout): Dropout(p=0.1, inplace=False)
        # )
        self.encoder = BertForMaskedLM.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        ).bert.embeddings

    def forward(self, inputs):
        return (
            self.encoder(inputs.input_ids).permute(1, 0, 2),
            1.0 - inputs.attention_mask.float(),
        )


class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        model = BertForMaskedLM.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        )
        self.encoder = model.bert

    def forward(self, inputs):
        output = self.encoder(**inputs)
        return output.last_hidden_state.permute(1, 0, 2), 1.0 - inputs.attention_mask

    def forward_full(self, inputs):
        output = self.encoder(**inputs)
        return (
            output.hidden_states[-3].permute(1, 0, 2),
            output.last_hidden_state.permute(1, 0, 2),
            1.0 - inputs.attention_mask,
        )
