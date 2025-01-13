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
import logging

import torch
import torch.nn as nn
from transformers import BertTokenizer

from network.layers.BERT_Modules import BERTNC, Teacher
from network.layers.Bert_ScoreAwareModels import BertEncoder as ScoreAwareBertEncoder
from network.layers.NN_Modules import Attention
from network.layers.Speech_Modules import Listener

# Getting logger information
logger = logging.getLogger("WCN-to-Text-Alignment")
TOK_NC = BertTokenizer.from_pretrained("bert-base-uncased")


class SLUModel(nn.Module):
    def __init__(self, config):
        super(SLUModel, self).__init__()
        if config["feats_type"] == "LFB":
            self.listener = Listener(
                config["input_dim"],
                config["pyr_layer"],
                config["device"],
                config["nlayer"],
                config["nhead"],
                enc_type=config["enc_type"],
                dropout=config["dropout"],
            )
        elif config["feats_type"] == "WCN":
            # Variables for the WCN encoder hardcoded values
            self.bert_model_name = "bert-base-uncased"
            self.fix_bert_model = False
            self.bert_dropout = 0.1
            self.bert_num_layers = config["wcn_nlayers"]
            self.bert_num_attn_heads = config["wcn_nheads"]
            self.sent_repr = "bin_sa_cls"  # taken from the original code
            self.score_util = "pp"  # taken from original code
            self.bert_tokenizer = config["bert_tokenizer"]
            self.bert_model = config["bert_model"]

            self.bert_model_opts = {
                "model": self.bert_model,
                "fix": self.fix_bert_model,
                "model_name": self.bert_model_name,
                "dp": self.bert_dropout,
            }
            self.bert_config = self.bert_model_opts["model"].config
            # For encoding WCN using scores as well
            self.utt_sa_bert_encoder = ScoreAwareBertEncoder(
                self.bert_config,
                self.bert_model_opts,
                self.score_util,
                n_layers=self.bert_num_layers,
                n_heads=self.bert_num_attn_heads,
            )
        self.sw_reader = BERTNC()
        # Cross attention Layer
        self.cross_attn = Attention(768, config["nhead"], dropout=config["dropout"])

        self.cls_layer = nn.Linear(768, config["nclasses"])
        self.dropout = nn.Dropout(config["dropout"])

        self.device = config["device"]
        self.feats_type = config["feats_type"]
        self.teacher = Teacher()
        logger.info("IC model initialized!")

    def encode_wcn_seq(self, inputs, masks):
        # inputs
        model_inputs_utt_sa = inputs
        self_mask = masks
        # encoder
        enc_out = self.utt_sa_bert_encoder(
            model_inputs_utt_sa, attention_mask=self_mask, default_pos=False
        )  # enc_out -> (bsz, seq_len, dm)
        return enc_out.permute(1, 0, 2)  # enc_out -> (seq_len, bsz, dm)

    def forward(self, input_s, input_smask, is_train=False):
        if self.feats_type == "LFB":
            speech, mask_s, _, _ = self.listener(input_s, is_train=is_train)
            bsz = len(input_s)
        elif self.feats_type == "WCN":
            wcn_mask = input_smask["self_mask"]
            speech = self.encode_wcn_seq(input_s, wcn_mask)
            mask_s = abs(input_s["mask"] - 1)
            bsz = input_s["tokens"].size(0)
        # QUERY is just CLS tokens as we aaume we do not have the text
        query = TOK_NC(
            ["[CLS]" for _ in range(bsz)],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        query, _ = self.sw_reader(query)
        query = query[0].unsqueeze(0)  # [1,bs,dim]  -> we keep the CLS encoding
        speech_cls, _ = self.cross_attn(query, speech, mask_s.bool())
        return self.cls_layer(self.dropout(speech_cls.squeeze(0)))


def extract(tens, mask, offset=0):
    out_lens = (1 - mask).sum(dim=1).tolist()
    out = []
    for i, ten in enumerate(tens):
        out.append(ten[: out_lens[i] - offset])
    return torch.cat(out, dim=0)
