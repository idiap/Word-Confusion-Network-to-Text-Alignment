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

The WCN encoder was inspired by the paper from Liu, C., Zhu, S., Zhao, Z., Cao, R., Chen, L., & Yu, K. (2020). Jointly Encoding Word Confusion Network and Dialogue Context with BERT for Spoken Language Understanding.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.layers.attention import SimpleSelfAttention
from network.layers.BERT_Modules import BERTNC, Teacher
from network.layers.Bert_ScoreAwareModels import BertEncoder as ScoreAwareBertEncoder
from network.layers.NN_Modules import Attention

# Internal libraries
from network.layers.Speech_Modules import Listener

# Getting logger information
logger = logging.getLogger("WCN-to-Text-Alignment")


class PTModel(nn.Module):
    def __init__(self, config):
        super(PTModel, self).__init__()
        # Only two type of representations: LFB or WCN -based
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
            # Following variables are needed for the WCN encoder
            self.bert_model_name = "bert-base-uncased"
            self.fix_bert_model = False
            self.bert_dropout = 0.1
            self.bert_num_layers = config["wcn_nlayers"]
            self.bert_num_attn_heads = config["wcn_nheads"]
            self.sent_repr = "bin_sa_cls"  # taken from the original code
            self.score_util = "pp"  # taken from original code
            # Loading BERT pretrain values again
            self.bert_tokenizer = config["bert_tokenizer"]
            self.bert_model = config["bert_model"]

            self.bert_model_opts = {
                "model": self.bert_model,
                "fix": self.fix_bert_model,
                "model_name": self.bert_model_name,
                "dp": self.bert_dropout,
            }
            self.bert_config = self.bert_model_opts["model"].config
            if self.sent_repr in ["bin_lstm", "bin_sa_cls", "tok_sa_cls"]:
                config["input_dim"] *= 2
            if self.sent_repr in ["attn", "bin_sa", "bin_sa_cls", "tok_sa_cls"]:
                self.slf_attn = SimpleSelfAttention(768, config["dropout"], "cuda")
            # For encoding WCN using scores as well
            self.utt_sa_bert_encoder = ScoreAwareBertEncoder(
                self.bert_config,
                self.bert_model_opts,
                self.score_util,
                n_layers=self.bert_num_layers,
                n_heads=self.bert_num_attn_heads,
            )

        self.sw_reader = BERTNC()
        self.cross_attn = Attention(768, config["nhead"], dropout=config["dropout"])
        self.teacher = Teacher()
        self.dropout = nn.Dropout(config["dropout"])
        self.feats_type = config["feats_type"]
        logger.info("\nPretrain model initialized!")

    def encode_wcn_seq(self, inputs, masks):
        model_inputs_utt_sa = inputs
        self_mask = masks
        enc_out = self.utt_sa_bert_encoder(
            model_inputs_utt_sa, attention_mask=self_mask, default_pos=False
        )
        # Returned tensor: enc_out -> (seq_len, bsz, dm)
        return enc_out.permute(1, 0, 2)

    def forward(self, input_s, input_smask, input_t, input_t_raw, is_train=False):
        if self.feats_type == "LFB":
            speech, mask_s, _, _ = self.listener(input_s, is_train=is_train)
        elif self.feats_type == "WCN":
            wcn_mask = input_smask["self_mask"]
            speech = self.encode_wcn_seq(input_s, wcn_mask)
            mask_s = abs(input_s["mask"] - 1)
        ncon_word, mask_t = self.sw_reader(input_t)
        con_word, attn = self.cross_attn(ncon_word, speech, mask_s.bool())
        # Theacher model is frozen
        with torch.no_grad():
            oracle, mask_t2 = self.teacher(input_t)

        speech_rep = extract(con_word.permute(1, 0, 2), mask_t.long())
        bert_rep = extract(oracle.permute(1, 0, 2), mask_t.long())

        return speech_rep, bert_rep, attn


def extract(tens, mask, offset=0):
    out_lens = (1 - mask).sum(dim=1).tolist()
    out = []
    for i, ten in enumerate(tens):
        out.append(ten[: out_lens[i] - offset])
    return torch.cat(out, dim=0)


class ContrastiveLoss(nn.Module):
    def __init__(self, device, temp=0.07):
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.temp = temp

    # FORWARD
    def forward(self, r1, r2):  # (bsz, dim=768)
        # r1 - speech representation
        # r2 - text representation
        r1 = F.normalize(r1, dim=1)
        r2 = F.normalize(r2, dim=1)
        assert r1.size(0) == r2.size(0)
        tgt = torch.eye(r1.size(0)).to(self.device)

        align = torch.matmul(r1, r2.t()) / self.temp
        al_0 = torch.log_softmax(align, dim=0)
        al_1 = torch.log_softmax(align, dim=1)

        loss_0 = -1.0 * self.temp * (al_0 * tgt).sum(dim=0).mean()
        loss_1 = -1.0 * self.temp * (al_1 * tgt).sum(dim=1).mean()
        loss = 0.5 * (loss_0 + loss_1)
        return loss
