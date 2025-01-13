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
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, classification_report, f1_score
from speechbrain.processing.features import InputNormalization
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import data.Constants as Constants
from network.layers.bert_xlnet_inputs import prepare_inputs_for_bert_xlnet
from network.layers.knn import kNN
from network.PretrainModel import ContrastiveLoss, PTModel
from network.TrainSLUModel import SLUModel

# Getting logger information
logger = logging.getLogger("WCN-to-Text-Alignment")


def get_global_group():
    if dist.is_initialized():
        if not hasattr(get_global_group, "_global_group"):
            get_global_group._global_group = dist.new_group()
        return get_global_group._global_group
    else:
        return None


def list_batch(X, lens, lmax):
    idx = list(range(len(lens)))
    random.shuffle(idx)
    sbatch_ = []
    for i, l in enumerate(lens):
        sbatch_.append(X[i, :l, :])
    return sbatch_


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


def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    logger.info("Validating dicts..")
    if len(model_state_dict_1) != len(model_state_dict_2):
        logger.info(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        logger.info(model_state_dict_1.keys() - model_state_dict_2.keys())
        return False
    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for (k_1, v_1), (k_2, v_2) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            logger.info(f"Key mismatch: {k_1} vs {k_2}")

        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            logger.info(f"Tensor mismatch: {v_1} vs {v_2}")


def load_dict(model, dict_path, isDPP=False):
    pretrained_dict = torch.load(dict_path, map_location="cpu")
    model_dict = model.state_dict()
    new_dict_from_pretrained = OrderedDict()
    for k, v in pretrained_dict.items():
        if k.startswith("module"):
            k = k.replace("module.", "")
        if k in model_dict:
            new_dict_from_pretrained.update({k: v})
        else:
            logger.info(f"key not in destination model {k}")
    assert (
        len(new_dict_from_pretrained) != 0
    ), f"No weights were loaded from the pretrained model {dict_path}"
    model_dict.update(new_dict_from_pretrained)
    model.load_state_dict(model_dict)
    return model


def load_slu_model(model, path, rank):
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


# Following three methods are required for masking the WCN
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


class Trainer(object):
    def __init__(
        self,
        tr_modality="",
        features_type="LFB",
        text_model_name=None,
        dataset_obj=None,
        trainDL=None,
        devDL=None,
        testDL=None,
        device="cpu",
        isDPP=False,
        run_folder="./",
        vocab_size=300,
        patience=10,
        learning_rate=0.0001,
        rank=None,
    ):

        self.modality = tr_modality
        self.features_type = features_type
        self.output_folder = run_folder
        self.device = device
        self.isDPP = isDPP

        self.model = None
        self.input_dim = 0
        self.nhead = 1
        self.dropout = 0.0
        self.lr = learning_rate
        self.rpat = patience
        self.nsteps = 600000

        self.dataset_obj = dataset_obj
        self.train_loader = trainDL
        self.devel_loader = devDL
        self.test_loader = testDL
        self.vocab_size = vocab_size

        # Required for the speech encoder LFB modality
        # Values are hard-coded, and are the same as those used in https://github.com/vishalsunder/Tokenwise-contrastive-pretraining
        self.pyr_layer = 3
        self.nlayer = 5
        self.enc_type = "lstm"

        # Required for WCNs encoding
        if self.features_type == "WCN":
            self.bert_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            self.bert_model = AutoModel.from_pretrained(text_model_name)
            # Addign special tokens
            self.bert_tokenizer.add_tokens(["<eps>", "<unk>"], special_tokens=True)
            self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))
        else:
            self.bert_tokenizer = None
            self.bert_model = None

        # Loss funciton used for pre-training mode
        self.con_loss = ContrastiveLoss(device)

        # Loss function used for fine-tuning to IC-SLU
        self.cls_loss = nn.CrossEntropyLoss()

        self.norm = InputNormalization(update_until_epoch=3)
        self.knn_moniter = kNN(
            self.devel_loader,
            self.device,
            self.norm,
            self.features_type,
            self.bert_tokenizer,
        )
        self.knn_evaluator = kNN(
            self.test_loader,
            self.device,
            self.norm,
            self.features_type,
            self.bert_tokenizer,
        )
        self.optimizer = None

        self.writer = SummaryWriter(os.path.join(run_folder, "summary"))
        self.model_path = os.path.join(run_folder, "models")
        # Path to the existing Models/ where the model will be saved
        if not os.path.exists(self.model_path) and int(rank) == 0:
            os.makedirs(self.model_path)

        self.logging_path = os.path.join(run_folder, "logging")
        if not os.path.exists(self.logging_path) and int(rank) == 0:
            os.makedirs(self.logging_path)
        # CHECKPOINTING
        self.checkpoint_path = os.path.join(run_folder, "checkpoints")
        if not os.path.exists(self.checkpoint_path) and int(rank) == 0:
            os.makedirs(self.checkpoint_path)

    def count_parameters(self, model) -> int:
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in list(
            filter(lambda p: p[1].requires_grad, model.named_parameters())
        ):
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        return total_params

    def initialize(
        self,
        acoustic_dim=768,
        nhead=12,
        nlayer=6,
        wcn_nlayers=12,
        wcn_nheads=12,
        dropout=0.1,
    ) -> None:
        self.input_dim = acoustic_dim
        self.nhead = nhead
        self.dropout = dropout
        self.nlayer = nlayer
        config = {
            "input_dim": 1 * self.input_dim,
            "device": self.device,
            "nhead": self.nhead,
            "dropout": self.dropout,
            "vocab_size": self.vocab_size,
            "pyr_layer": self.pyr_layer,
            "nlayer": self.nlayer,
            "enc_type": self.enc_type,
            "feats_type": self.features_type,
            "bert_tokenizer": self.bert_tokenizer,
            "bert_model": self.bert_model,
            "wcn_nlayers": wcn_nlayers,
            "wcn_nheads": wcn_nheads,
        }
        # Creating the pre-training object
        self.model = PTModel(config)
        # Counting the number of parameters in the model
        total_params = self.count_parameters(self.model)
        logger.info(f"Total Trainable Params: {total_params}")

    def initialize_for_finetuning(
        self,
        acoustic_dim=768,
        nhead=12,
        nlayer=6,
        wcn_nlayers=12,
        wcn_nheads=12,
        dropout=0.1,
        num_classes=10,
        eval=False,
    ) -> None:
        self.input_dim = acoustic_dim
        self.nhead = nhead
        self.dropout = dropout
        self.nlayer = nlayer
        config = {
            "input_dim": 1 * self.input_dim,
            "device": self.device,
            "nhead": self.nhead,
            "dropout": self.dropout,
            "vocab_size": self.vocab_size,
            "pyr_layer": self.pyr_layer,
            "nlayer": self.nlayer,
            "enc_type": self.enc_type,
            "nclasses": num_classes,
            "feats_type": self.features_type,
            "bert_tokenizer": self.bert_tokenizer,
            "bert_model": self.bert_model,
            "wcn_nlayers": wcn_nlayers,
            "wcn_nheads": wcn_nheads,
        }
        self.model = SLUModel(config)
        total_params = self.count_parameters(self.model)
        logger.info(f"Total Trainable Params: {total_params}")

    # Saving models
    def save(self, model, path, rank=0):
        if int(rank) == 0:
            logger.info(f"\n Saving model as: {path}")
            torch.save(model.state_dict(), path)

    # Saving models
    def save_SLU_model(self, model, path, rank=0):
        if int(rank) == 0:
            logger.info(f"\n Saving model as: {path}")
            torch.save(model.module.state_dict(), path)

    # CHECKPOINTING - saving checkpoint
    def checkpoint_model(
        self,
        fname,
        model_state_dict,
        epoch,
        steps,
        optimizer_state_dict,
        rpat,
        best_score,
        best_model,
        loss_list,
        scores_val,
        rank=0,
    ):
        if int(rank) == 0:
            logger.info("SAVING CHECKPOINT.")
            logger.info(
                "Saving as:{}".format(os.path.join(self.checkpoint_path, fname))
            )
            state = {
                "model": model_state_dict,
                "epoch": epoch,
                "steps": steps,
                "optimizer": optimizer_state_dict,
                "rpat": rpat,
                "best_score": best_score,
                "bets_model": best_model,
                "loss_list": loss_list,
                "scores_val": scores_val,
            }
            torch.save(state, os.path.join(self.checkpoint_path, fname))

    # CHECKPOINTING - load checkpoint
    def load_checkpoint(self, chkpt_name=""):
        chkpt = torch.load(
            os.path.join(self.checkpoint_path, chkpt_name), map_location="cpu"
        )
        state_dict = chkpt["model"]
        epoch = chkpt["epoch"]
        steps = chkpt["steps"]
        optimizer = chkpt["optimizer"]
        rounds = chkpt["rpat"]
        best_score = chkpt["best_score"]
        best_model = chkpt["bets_model"]
        loss_list = chkpt["loss_list"]
        scores_val = chkpt["scores_val"]
        return (
            state_dict,
            epoch,
            steps,
            optimizer,
            rounds,
            best_score,
            best_model,
            loss_list,
            scores_val,
        )

    def opt_step(self, model, loss):
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        self.optimizer.step()

    def opt_step_dpp(self, model, loss, world_size):
        loss.backward()
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        self.optimizer.step()

    def perform_validation(self, rank, model, steps):
        logger.info(f"\n\n==> Running validation when DPP in rank: {rank} <==")
        score_val = self.knn_moniter.run_alt(model)
        if int(rank) == 0:
            self.writer.add_scalar("Accuracy/valid", score_val, steps)
        logger.info(f"\n| steps = {steps} | dev_acc = {score_val} | rank = {rank} |")
        return score_val

    def perform_slu_validation(self, rank, model):
        """
        This method evaluates the accuracy of the alignments using the run_alt version of KNN.
        Using batches and averaging the results
        """
        logger.info(f"\n\n==> Running validation when DPP in rank: {rank} <==")
        score_val = self.knn_evaluator.run_alt_for_slu(model)
        logger.info(f"\n| dev_acc = {score_val} | rank = {rank} |")
        return score_val

    def perform_evaluation(self, rank, model):
        logger.info(f"\n\n==> Running evaluation when DPP in rank: {rank} <==")
        score_eval = self.knn_evaluator.run_alt(
            model
        )  # FIXME This version works if DPP is True
        logger.info(
            f"\n| Evaluation ACCURACY\n |  mean test_acc = {score_eval} | rank = {rank} |"
        )
        return score_eval

    def pretrain(
        self,
        validate_after=2000,
        checkpoint_after=1000,
        log_after=100,
        world_size=0,
        save_after=100000,
        save_model=True,
    ):
        assert self.model is not None
        steps = 0
        best_score = None
        best_model = None
        epoch = 0
        loss_list = []
        scores_val = []
        rpat = self.rpat
        chkpt = False
        if self.isDPP:
            rank = dist.get_rank()
            logger.info(f"RANK:{rank}")
        else:
            rank = 0  # Making sure the rank variable to be set to 0 in case DPP==False

        if int(rank) == 0:
            logger.info("\n=== Pretraining Speech-To-BERT alignment ===")

        self.model.to(self.device)
        if self.isDPP:
            distributed_model = DistributedDataParallel(
                self.model, find_unused_parameters=True
            )
        else:
            distributed_model = self.model
            distributed_model.to(self.device)
        # Validating Check point exists
        if os.path.exists(
            os.path.join(self.checkpoint_path, "latest_ckpt_training.pt")
        ):
            logger.info("\t ==> Starting from Checkpoint <==")
            (
                last_state_dict,
                last_epoch,
                last_step,
                last_optimizer_state_dict,
                last_rpat,
                last_best_score,
                last_best_model,
                last_loss_list,
                last_scores_val,
            ) = self.load_checkpoint(chkpt_name="latest_ckpt_training.pt")
            chkpt = True
            self.model.load_state_dict(last_state_dict)
            epoch = last_epoch
            steps = last_step
            rpat = last_rpat
            best_score = last_best_score
            best_model = last_best_model
            logger.info(
                f"\nStarting from epoch:{epoch}, steps:{steps}, patience:{rpat}, previous best score:{best_score}\n"
            )

        self.optimizer = AdamW(
            distributed_model.parameters(),
            lr=self.lr,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=0,
        )
        if chkpt is True:
            self.optimizer.load_state_dict(last_optimizer_state_dict)
        while 600000 > steps:  # Pretraining is done up to 600K steps
            epoch = epoch + 1
            # Sampler object -- if DPP, we make the shuffling work properly across multiple epochs and multiple GPUs.
            if self.isDPP:
                self.dataset_obj.get_train_sampler().set_epoch(epoch)

            if int(rank) == 0:
                logger.info(f"Running epoch {epoch}.")
            for i, batch in enumerate(
                tqdm(self.train_loader, desc="PeoplesSpeech 1Khrs (Train) BATCH_NUMS")
            ):
                distributed_model.train()
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
                steps += world_size * 1
                if self.features_type == "LFB":
                    lens_norm = [1.0 * (x / lmax) for x in lens]
                    sbatch = self.norm(
                        X, torch.tensor(lens_norm).float(), epoch=epoch - 1
                    )
                    sbatch = list_batch(sbatch, lens, lmax)
                    sbatch = load2gpu(sbatch, self.device)
                elif self.features_type == "WCN":
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
                self.optimizer.zero_grad()
                # FORWARD PASS
                r_s, r_b, _ = distributed_model(sbatch, smask, tbatch, tbatch_raw)
                loss = self.con_loss(r_s, r_b.detach())
                if rank == 0:
                    self.writer.add_scalar("Loss/train", loss, steps)
                if self.isDPP:
                    self.opt_step_dpp(distributed_model, loss, world_size)
                else:
                    self.opt_step(distributed_model, loss)
                loss_list.append(loss.item())
                if steps % save_after == 0:
                    self.save(
                        distributed_model,
                        os.path.join(self.model_path, f"model_steps_{steps}.pt"),
                        rank,
                    )
                if steps % checkpoint_after == 0:
                    # Checkpointing for distributed learning
                    if self.isDPP:
                        self.checkpoint_model(
                            "latest_ckpt_training.pt",
                            distributed_model.module.state_dict(),
                            epoch,
                            steps,
                            self.optimizer.state_dict(),
                            rpat,
                            best_score,
                            best_model,
                            loss_list,
                            scores_val,
                            rank,
                        )
                    else:
                        self.checkpoint_model(
                            "latest_ckpt_training.pt",
                            distributed_model.state_dict(),
                            epoch,
                            steps,
                            self.optimizer.state_dict(),
                            rpat,
                            best_score,
                            best_model,
                            loss_list,
                            scores_val,
                            rank,
                        )

                if steps % validate_after == 0:
                    # if self.isDPP:
                    score_val = self.perform_validation(rank, distributed_model, steps)
                    scores_val.append(score_val)
                    logger.info("\nFinalized the knn validation\n")

                    if best_score is None or best_score < score_val:
                        logger.info(f"Saving intermediate model as {self.model_path}")
                        self.save(
                            distributed_model,
                            os.path.join(
                                self.model_path, "Current_Best_Inter_Model.pt"
                            ),
                            rank,
                        )
                        best_score = score_val
                        rpat = self.rpat
                    else:
                        rpat -= 1
                if steps % log_after == 0:
                    logger.info(
                        f"| steps = {steps} | loss = {np.mean(loss_list)} | patience = {rpat} |"
                    )
                    loss_list = []

        if save_model:
            self.save(
                distributed_model.to("cpu"),
                os.path.join(self.model_path, "Final_model.pt"),
                rank,
            )
        self.writer.flush()

        logger.info("\n==> DONE PRETRAINING <==\n")
        score_val = self.perform_validation(rank, distributed_model, steps)
        _ = self.perform_evaluation(rank, distributed_model.to(self.device))
        logger.info("\n==> EVALUATING <==\n")

    def evaluate_pretraining(self, model_dict_path=""):
        assert self.model is not None
        if self.isDPP:
            rank = dist.get_rank()
            logger.info(f"RANK:{rank}")
        else:
            rank = 0
        assert (
            model_dict_path != ""
        ), "No pretrained model was provided in 'model_dict' parameter"
        # Loading pretrained weights
        self.model = load_dict(self.model, model_dict_path)
        self.model.to(self.device)
        logger.info(f"Evaluating alignments with model: {model_dict_path}")
        if self.isDPP:
            distributed_model = DistributedDataParallel(
                self.model, find_unused_parameters=True
            )
        else:
            distributed_model = self.model
            distributed_model.to(self.device)
        logger.info("\n==> EVALUATING <==\n")
        _ = self.perform_validation(rank, distributed_model, 600000)
        _ = self.perform_evaluation(rank, distributed_model)

    def evaluate_alignment_of_slu_model(self, model_dict_path=""):
        assert self.model is not None
        if self.isDPP:
            rank = dist.get_rank()
            logger.info(f"RANK:{rank}")
        else:
            rank = 0
        assert (
            model_dict_path != ""
        ), "No pretrained model was provided in 'model_dict' parameter"
        # Loading weights from finetuned IC model
        self.model = load_dict(self.model, model_dict_path)
        self.model.to(self.device)
        logger.info(f"Evaluating alignments with model: {model_dict_path}")
        if self.isDPP:
            distributed_model = DistributedDataParallel(
                self.model, find_unused_parameters=True
            )
        else:
            distributed_model = self.model
            distributed_model.to(self.device)

        logger.info("\n==> EVALUATING <==\n")
        _ = self.perform_slu_validation(rank, distributed_model)
        return None

    def evaluate_slu(self, model, loader, rank=0, epochs=0, test=False):
        model.eval()
        y_pred = []
        y_true = []
        for i, batch in enumerate(tqdm(loader, desc="SLURP (Devel/Test) BATCH_NUMS")):
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
                label,
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
                batch["labels_batch_raw"],
            )

            if self.features_type == "LFB":
                lens_norm = [1.0 * (x / lmax) for x in lens]
                sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=epochs - 1)
                sbatch = list_batch(sbatch, lens, lmax)
                sbatch = load2gpu(sbatch, self.device)
            elif self.features_type == "WCN":  # Working with WCNs
                wcn_lens = [len(utt) for utt in wcn_in_seqs]
                # Preparing the input of the WCN values
                # Sbatch aquires a different meaning for WCN encoding
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
            label = label.to(self.device)
            tbatch, tbatch_raw = load2gpu(tbatch, self.device), load2gpu(
                tbatch_raw, self.device
            )
            with torch.no_grad():
                pred = model(sbatch, smask)
            y_pred.extend(torch.max(pred, dim=1)[1].cpu().tolist())
            y_true.extend(label.cpu().tolist())
        f1_score_val = f1_score(y_true, y_pred, average="macro")
        acc_score = accuracy_score(y_true, y_pred)

        if int(rank) == 0 and test is False:
            self.writer.add_scalar("F1/valid", f1_score_val, epochs)
            self.writer.add_scalar("Accuracy/valid", acc_score, epochs)
        if int(rank) == 0 and test is True:
            logger.info(classification_report(y_true, y_pred))
        return f1_score_val, acc_score

    def slu_finetuning(self, checkpoint_after=100, model_dict_path="", world_size=1):

        assert self.model is not None
        assert (
            model_dict_path != ""
        ), "No pretrained model was provided in 'model_dict' parameter"
        self.model = load_dict(self.model, model_dict_path)
        best_score = None
        best_model = None
        epoch = 0
        steps = 0
        loss_list = []
        rpat = self.rpat
        chkpt = False
        if self.isDPP:
            rank = dist.get_rank()
            logger.info(f"RANK:{rank}")
        else:
            rank = 0

        if int(rank) == 0:
            logger.info("\n=== Finetuing for SLU task ===")
        self.model.to(self.device)
        if self.isDPP:
            # DDP makes sure the model is the same across all devices
            if self.features_type == "LFB" or self.features_type == "WCN":
                distributed_model = DistributedDataParallel(
                    self.model, find_unused_parameters=True
                )
        else:
            distributed_model = self.model

        # Checking for checkpoint
        if os.path.exists(
            os.path.join(self.checkpoint_path, "latest_ckpt_slu_training.pt")
        ):
            logger.info("\t ==> Starting from Checkpoint <==")
            (
                last_state_dict,
                last_epoch,
                last_step,
                last_optimizer_state_dict,
                last_rpat,
                last_best_score,
                last_best_model,
                last_loss_list,
                last_scores_val,
            ) = self.load_checkpoint(chkpt_name="latest_ckpt_slu_training.pt")
            chkpt = True
            self.model.load_state_dict(last_state_dict)
            epoch = last_epoch
            steps = last_step
            rpat = last_rpat
            best_score = last_best_score
            best_model = last_best_model
            logger.info(
                f"\nStarting from epoch:{epoch}, steps:{steps}, patience:{rpat}, previous best score:{best_score}\n"
            )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=0,
        )
        if chkpt is True:
            self.optimizer.load_state_dict(last_optimizer_state_dict)
        while rpat > 0:
            epoch = epoch + 1
            if self.isDPP:
                self.dataset_obj.get_train_sampler().set_epoch(epoch)
            if int(rank) == 0:
                logger.info(f"Running epoch {epoch}.")

            for i, batch in enumerate(
                tqdm(self.train_loader, desc="(Train) BATCH_NUMS")
            ):
                steps = steps + 1
                distributed_model.train()
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
                    label,
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
                    batch["labels_batch_raw"],
                )

                if self.features_type == "LFB":
                    lens_norm = [1.0 * (x / lmax) for x in lens]
                    sbatch = self.norm(
                        X, torch.tensor(lens_norm).float(), epoch=epoch - 1
                    )
                    sbatch = list_batch(sbatch, lens, lmax)
                    sbatch = load2gpu(sbatch, self.device)
                elif self.features_type == "WCN":
                    wcn_lens = [len(utt) for utt in wcn_in_seqs]
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
                label = label.to(self.device)
                self.optimizer.zero_grad()
                pred = distributed_model(sbatch, smask)
                loss = self.cls_loss(pred, label)
                if int(rank) == 0:
                    self.writer.add_scalar("Loss/train", loss, steps)
                if self.isDPP:
                    self.opt_step_dpp(distributed_model, loss, world_size)
                else:
                    self.opt_step(distributed_model, loss)
                loss_list.append(loss.item())
            logger.info("\n --> Running validation <--")
            f1_score_val, acc_score = self.evaluate_slu(
                distributed_model,
                self.devel_loader,
                rank=rank,
                epochs=epoch,
                test=False,
            )
            f1_score_val_tensor = torch.tensor(f1_score_val).to(self.device)
            dist.all_reduce(f1_score_val_tensor)
            acc_score_tensor = torch.tensor(acc_score).to(self.device)
            dist.all_reduce(acc_score_tensor)

            average_f1_score = f1_score_val_tensor.item() / world_size
            average_accuracy = acc_score_tensor / world_size

            if best_score is None or best_score < average_f1_score:
                logger.info(f"saving intermediate model as {self.model_path}")
                self.save_SLU_model(
                    distributed_model,
                    os.path.join(self.model_path, "Current_Best_SLU_Model.pt"),
                    rank=rank,
                )
                best_score = average_f1_score
                rpat = self.rpat
            else:
                rpat -= 1
            logger.info(
                f"\n| epoch = {epoch} | loss = {np.mean(loss_list)} | dev_f1_score = {f1_score_val} | dev_acc_score = {acc_score} | rank = {rank}"
            )
            if rank == 0:
                logger.info(
                    f"\n| epoch = {epoch} | loss = {np.mean(loss_list)} | dev_f1_score = {average_f1_score} | dev_acc_score = {average_accuracy} | rank = {rank}"
                )
            if epoch % checkpoint_after == 0:
                if self.isDPP:
                    self.checkpoint_model(
                        "latest_ckpt_slu_training.pt",
                        distributed_model.module.state_dict(),
                        epoch,
                        steps,
                        self.optimizer.state_dict(),
                        rpat,
                        best_score,
                        best_model,
                        loss_list,
                        [],
                        rank=rank,
                    )
                else:
                    self.checkpoint_model(
                        "latest_ckpt_slu_training.pt",
                        distributed_model.state_dict(),
                        epoch,
                        steps,
                        self.optimizer.state_dict(),
                        rpat,
                        best_score,
                        best_model,
                        loss_list,
                        [],
                        rank=rank,
                    )
            loss_list = []
        logger.info("\n --> Running test <--")
        if rank == 0:
            best_model_name = os.path.join(self.model_path, "Current_Best_SLU_Model.pt")
            logger.info(f"\t Reading file {best_model_name}")

            best_model = load_slu_model(self.model, best_model_name, rank)
            best_model.to(self.device)

            f1_score_test, acc_score_test = self.evaluate_slu(
                best_model, self.test_loader, rank=rank, epochs=epoch, test=True
            )
            logger.info(
                f"| Test F1 score = {f1_score_test} | Test Acc score = {acc_score_test}|"
            )
        logger.info("Done")
        return 0
