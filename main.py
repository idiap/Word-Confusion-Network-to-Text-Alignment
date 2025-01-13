#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2024> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Esau Villatoro-Tello <esau.villatoro@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

"""
Script implementing a simple Word-Confusion-Network-to-Text Alignment Approach for Intent Classification
"""

import logging
import os
import random
import sys

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

import data.dataset as data
import network.Trainer as trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("WCN-to-Text-Alignment")


def setup_dist(
    rank, world_size, master_port=None, use_ddp_launch=False, master_addr=None
) -> None:
    """
    rank: rank number for the current GPU
    worls_size: maximum number of GPUs to be used
    master_port: Master port value
    use_dpp_launch: boolean value that indicates to use distributed learning
    master_addr: master port address

    Note: rank and world_size are used only if use_ddp_launch is True
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = (
            "localhost" if master_addr is None else str(master_addr)
        )
    logger.info(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12354" if master_port is None else str(master_port)

    if use_ddp_launch is False:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        dist.init_process_group("nccl")


def cleanup_dist():
    """Destroy the distributed process"""
    dist.destroy_process_group()


def run(rank, world_size, args):
    # Setting the random seed manually for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Getting the number of jobs/GPUs to be allocated
    world_size = world_size
    logger.info(f"World_Size: {world_size}")

    # Launching distributed DDP
    if args.distributed is True:
        master_port = None
        if int(rank) == 0:
            logger.info(
                f"Calling setup_dist with rank= {rank} world_size={world_size} master_port={master_port}"
            )
        setup_dist(rank, world_size, master_port)
        if int(rank) == 0:
            logger.info("Distributed INIT passed !!")
    else:
        logger.info("Running single GPU/CPU !!")
    # If CUDA is not available.  Note: be aware the process will be really slow if run in CPU
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        if int(rank) == 0:
            logger.info(f"Running on GPU, device: {device}")
    else:
        device = torch.device("cpu")

    # pre-training step
    if args.mode == "pretrain" or args.mode == "eval_pretrain":
        if args.train_set is not None and args.test_set is not None:
            if int(rank) == 0:
                logger.info("Loading Data")

            d = data.WCN2B_Dataset(
                dataset_name=args.dataset_name,
                test_dataset_name=args.test_dataset_name,
                train=args.train_set,
                dev=args.dev_set,
                test=args.test_set,
                test_audios=args.test_set_audios,
                isDPP=args.distributed,
                txt_model=args.text_model,
                features_type=args.acoustic_feats_type,
                acou_dimension=args.acoustic_dim,
                rank=rank,
                world_size=world_size,
            )

            # Generating loader objects
            train_set, dev_set, test_set = d.generate_DataLoaderObject(
                bs=args.batch_size, output_folder=args.runs_folder
            )
            if int(rank) == 0:
                logger.info("DataLoaders correctly generated")
            vocab_size = d.get_vocabulary_size()

        # Creating Training Object
        pre_train_obj = trainer.Trainer(
            tr_modality=args.mode,
            features_type=args.acoustic_feats_type,
            text_model_name=args.text_model,
            dataset_obj=d,
            trainDL=train_set,
            devDL=dev_set,
            testDL=test_set,
            device=device,
            isDPP=args.distributed,
            run_folder=args.runs_folder,
            vocab_size=vocab_size,
            patience=args.patience,
            learning_rate=args.learning_rate,
            rank=rank,
        )
        # Initializing pre_train objects
        pre_train_obj.initialize(
            acoustic_dim=1 * args.acoustic_dim,
            nhead=args.number_heads,
            nlayer=args.number_layers,
            wcn_nlayers=args.wcn_num_of_layers,
            wcn_nheads=args.wcn_num_attn_heads,
            dropout=args.dropout,
        )
        # Starting the actual pretraining
        if args.mode == "pretrain":
            pre_train_obj.pretrain(
                validate_after=args.validate_after,
                log_after=args.log_after,
                world_size=world_size,
                save_after=args.save_after,
                save_model=args.save_model,
            )
        elif args.mode == "eval_pretrain":
            pre_train_obj.evaluate_pretraining(model_dict_path=args.pretrained_model)
    # ELSE --> Finetuning for Intent on SLURP or FSC
    elif args.mode == "slu_ft" or args.mode == "evaluate_slu_ft":
        if args.dataset_name == "SLURP":
            logger.info("Locating and reading SLURP data for SLU finetuning")

            d = data.SLURP_Dataset(
                train=args.train_set,
                dev=args.dev_set,
                test=args.test_set,
                test_audios=args.test_set_audios,
                isDPP=args.distributed,
                txt_model=args.text_model,
                features_type=args.acoustic_feats_type,
                acou_dimension=args.acoustic_dim,
                output_folder=args.runs_folder,
                clasification_task="intent",
                rank=rank,
                world_size=world_size,
                train_wcn=args.train_WCN_file,
                dev_wcn=args.dev_WCN_file,
                test_wcn=args.test_WCN_file,
            )
            # Generating loader objects
            train_set, dev_set, test_set = d.generate_DataLoaderObject(
                bs=args.batch_size
            )
            vocab_size = d.get_vocabulary_size()
            num_labels = d.get_num_classes()

        # Creating Training Object and fine-tuning
        finetunning_obj = trainer.Trainer(
            tr_modality=args.mode,
            features_type=args.acoustic_feats_type,
            text_model_name=args.text_model,
            dataset_obj=d,
            trainDL=train_set,
            devDL=dev_set,
            testDL=test_set,
            device=device,
            isDPP=args.distributed,
            run_folder=args.runs_folder,
            vocab_size=vocab_size,
            patience=args.patience,
            learning_rate=args.learning_rate,
            rank=rank,
        )

        eval_alignments = True if args.mode == "evaluate_slu_ft" else False
        finetunning_obj.initialize_for_finetuning(
            acoustic_dim=args.acoustic_dim,
            nhead=args.number_heads,
            nlayer=args.number_layers,
            wcn_nlayers=args.wcn_num_of_layers,
            wcn_nheads=args.wcn_num_attn_heads,
            dropout=args.dropout,
            num_classes=num_labels,
            eval=eval_alignments,
        )

        if args.mode == "slu_ft":
            finetunning_obj.slu_finetuning(
                checkpoint_after=args.checkpoint_after,
                model_dict_path=args.pretrained_model,
                world_size=world_size,
            )
        elif args.mode == "evaluate_slu_ft":
            finetunning_obj.evaluate_alignment_of_slu_model(
                model_dict_path=args.slu_model
            )
    if int(rank) == 0:
        logger.info("DONE ;)")


@hydra.main(config_path="./", config_name="config_pretrain.yaml")
def main(cfg: DictConfig) -> None:
    # Parsing arguments
    args = cfg
    logger.info("ARGUMENTS:")
    logger.info(OmegaConf.to_yaml(cfg))
    world_size = args.num_jobs
    assert world_size >= 1
    # If more than 1 GPU available
    if world_size > 1:
        mp.set_start_method("spawn")
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# Calling main method
if __name__ == "__main__":
    main()
