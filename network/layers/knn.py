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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import data.Constants as Constants
from network.layers.bert_xlnet_inputs import prepare_inputs_for_bert_xlnet


def load2gpu(x, device):
    if x is None:
        return x
    if isinstance(x, dict):
        t2 = {}
        for key, val in x.items():
            t2[key] = val.to(device)
        return t2
    if isinstance(x, list):
        y = []
        for v in x:
            y.append(v.to(device))
        return y
    return x.to(device)


def list_batch(X, lens, lmax):
    idx = list(range(len(lens)))
    random.shuffle(idx)
    sbatch_ = []
    for i, l in enumerate(lens):
        sbatch_.append(X[i, :l, :])
    return sbatch_


# Following three methods are required for masking the WCN'''
# TODO: These methods are repeated in TRAINER CLASS, need to fix\


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def prepare_wcn_mask(inputs):
    masks = {}
    tokens = inputs["tokens"]
    self_mask = get_attn_key_pad_mask(tokens, tokens)
    non_pad_mask = get_non_pad_mask(tokens)
    masks["self_mask"] = self_mask
    masks["non_pad_mask"] = non_pad_mask
    return masks


class kNN(object):
    def __init__(self, loader, device, norm, feats_type, tokenizer):
        self.loader = loader
        self.device = device
        self.norm = norm
        self.feats_type = feats_type
        self.bert_tokenizer = tokenizer

    def get_embd(self, model):
        model.eval()
        embd_s = []
        embd_b = []
        for i, batch in enumerate(tqdm(self.loader, desc="(Validation) BATCH_NUMS")):
            (
                X,
                lens,
                lmax,
                smask,
                tbatch,
                tbatch_raw,
                wcn_in_seqs,
                wcn_pos_seqs,
                wcn_score_seqs,
            ) = (
                batch["speech_feats"],
                batch["speech_lens"],
                batch["speech_max_len"],
                batch["speech_mask"],
                batch["textual_batch"],
                batch["text_batch_raw"],
                batch["in_seqs"],
                batch["pos_seqs"],
                batch["score_seqs"],
            )
            if self.feats_type == "LFB":
                lens_norm = [1.0 * (x / lmax) for x in lens]
                sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=100)
                sbatch = list_batch(sbatch, lens, lmax)
                sbatch = load2gpu(sbatch, self.device)
            elif self.feats_type == "WCN":
                wcn_lens = [len(utt) for utt in wcn_in_seqs]
                # Preparing the input of the WCN values
                # sbatch aquires a different meaning for WCN encoding
                sbatch = prepare_inputs_for_bert_xlnet(
                    wcn_in_seqs,
                    wcn_lens,
                    self.bert_tokenizer,
                    wcn_pos_seqs,
                    wcn_score_seqs,
                    cls_token_at_end=False,
                    cls_token="[CLS]",
                    sep_token="[SEP]",
                    cls_token_segment_id=0,
                    pad_on_left=False,
                    pad_token_segment_id=0,
                    device=torch.device(self.device),
                )
                smask = prepare_wcn_mask(sbatch)
            smask = load2gpu(smask, self.device)
            tbatch, tbatch_raw = load2gpu(tbatch, self.device), load2gpu(
                tbatch_raw, self.device
            )
            with torch.no_grad():
                s_rep, b_rep, _ = model(sbatch, smask, tbatch, tbatch_raw)
            embd_s.append(s_rep)
            embd_b.append(b_rep)
        return torch.cat(embd_s, dim=0), torch.cat(embd_b, dim=0)

    def run(self, model):
        print("Encoding dev set for NN search. Running run")
        s_rep, b_rep = self.get_embd(model)
        assert s_rep.size(0) == b_rep.size(
            0
        ), f"No. of speech embeddings ({s_rep.size(0)}) != No. of BERT embeddings ({s_rep.size(0)})"
        N = s_rep.size(0)
        s_rep_norm, b_rep_norm = (
            F.normalize(s_rep, dim=1).cpu(),
            F.normalize(b_rep, dim=1).cpu(),
        )
        y_true = list(range(N))
        align = torch.matmul(s_rep_norm, b_rep_norm.t())
        y_pred = torch.max(align, dim=1)[1].tolist()

        return accuracy_score(y_true, y_pred)

    def run_alt(self, model):
        print("Encoding dev set for NN search. Running run_alt\n")
        model.eval()
        acc_scores = []
        for i, batch in enumerate(tqdm(self.loader, desc=" (Test) BATCH_NUMS")):
            (
                X,
                lens,
                lmax,
                smask,
                tbatch,
                tbatch_raw,
                wcn_in_seqs,
                wcn_pos_seqs,
                wcn_score_seqs,
            ) = (
                batch["speech_feats"],
                batch["speech_lens"],
                batch["speech_max_len"],
                batch["speech_mask"],
                batch["textual_batch"],
                batch["text_batch_raw"],
                batch["in_seqs"],
                batch["pos_seqs"],
                batch["score_seqs"],
            )
            if self.feats_type == "LFB":
                lens_norm = [1.0 * (x / lmax) for x in lens]
                sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=100)
                sbatch = list_batch(sbatch, lens, lmax)
                sbatch = load2gpu(sbatch, self.device)
            elif self.feats_type == "WCN":
                wcn_lens = [len(utt) for utt in wcn_in_seqs]
                # Preparing the input of the WCN values
                # sbatch aquires a different meaning for WCN encoding
                sbatch = prepare_inputs_for_bert_xlnet(
                    wcn_in_seqs,
                    wcn_lens,
                    self.bert_tokenizer,
                    wcn_pos_seqs,
                    wcn_score_seqs,
                    cls_token_at_end=False,
                    cls_token="[CLS]",
                    sep_token="[SEP]",
                    cls_token_segment_id=0,
                    pad_on_left=False,
                    pad_token_segment_id=0,
                    device=torch.device(self.device),
                )
                smask = prepare_wcn_mask(sbatch)
            smask = load2gpu(smask, self.device)
            tbatch, tbatch_raw = load2gpu(tbatch, self.device), load2gpu(
                tbatch_raw, self.device
            )
            with torch.no_grad():
                s_rep, b_rep, _ = model(sbatch, smask, tbatch, tbatch_raw)
            assert s_rep.size(0) == b_rep.size(
                0
            ), f"No. of speech embeddings ({s_rep.size(0)}) != No. of BERT embeddings ({s_rep.size(0)})"
            N = s_rep.size(0)
            s_rep_norm, b_rep_norm = (
                F.normalize(s_rep, dim=1).cpu(),
                F.normalize(b_rep, dim=1).cpu(),
            )
            y_true = list(range(N))
            align = torch.matmul(s_rep_norm, b_rep_norm.t())
            y_pred = torch.max(align, dim=1)[1].tolist()
            local_acc = accuracy_score(y_true, y_pred)
            acc_scores.append(local_acc)
        return np.mean(np.array(acc_scores))

    def run_alt_for_slu(self, model):
        print("Encoding dev set for NN search. Running run_alt_for_slu\n")
        model.eval()
        acc_scores = []
        for i, batch in enumerate(tqdm(self.loader, desc="(Eval) BATCH_NUMS")):
            X, lens, lmax, smask, tbatch, tbatch_raw = (
                batch["speech_feats"],
                batch["speech_lens"],
                batch["speech_max_len"],
                batch["speech_mask"],
                batch["textual_batch"],
                batch["text_batch_raw"],
            )
            if self.feats_type == "LFB":
                lens_norm = [1.0 * (x / lmax) for x in lens]
                sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=100)
                sbatch = list_batch(sbatch, lens, lmax)
                sbatch = load2gpu(sbatch, self.device)
            smask = smask.to(self.device)
            tbatch, tbatch_raw = load2gpu(tbatch, self.device), load2gpu(
                tbatch_raw, self.device
            )
            with torch.no_grad():
                s_rep, b_rep = model.forward_for_eval_align(
                    sbatch, smask, tbatch, tbatch_raw
                )
            assert s_rep.size(0) == b_rep.size(
                0
            ), f"No. of speech embeddings ({s_rep.size(0)}) != No. of BERT embeddings ({s_rep.size(0)})"
            N = s_rep.size(0)
            s_rep_norm, b_rep_norm = (
                F.normalize(s_rep, dim=1).cpu(),
                F.normalize(b_rep, dim=1).cpu(),
            )
            y_true = list(range(N))
            align = torch.matmul(s_rep_norm, b_rep_norm.t())
            y_pred = torch.max(align, dim=1)[1].tolist()
            local_acc = accuracy_score(y_true, y_pred)
            acc_scores.append(local_acc)
        return accuracy_score(y_true, y_pred)
