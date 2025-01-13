#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2024> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Esau Villatoro-Tello <esau.villatoro@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import logging
import os
import re
import xml.etree.ElementTree as et
from typing import Tuple

import numpy as np
import pandas as pd

# Torch libraries
import torch
import torchaudio
import torchaudio.transforms as AT
from progress.bar import Bar
from sklearn import preprocessing
from speechbrain.processing.features import STFT, Filterbank, spectral_magnitude
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDatasetObj

# for pretrained models
from transformers import AutoModel, AutoTokenizer

# Getting logger information
logger = logging.getLogger("WCN-to-Text-Alignment")


def clean_str(text) -> str:
    text = re.sub(r"[^A-Za-z0-9\s]+", "", text)
    return text.lower().strip()


def get_mask(lens):
    mask = torch.ones(len(lens), max(lens))
    for i, l in enumerate(lens):
        mask[i][:l] = 0.0
    return mask


def pad(input, factor=4):
    add_size = input.size(0) % factor
    if add_size != 0:
        rem_size = factor - add_size
        return torch.cat([input, torch.zeros(rem_size, input.size(1))], dim=0)
    else:
        return input


def padding(sbatch) -> Tuple[torch.Tensor, list, int]:
    dim = sbatch[0].size(2)
    lens = [x.size(1) for x in sbatch]
    lmax = max(lens)
    padded = []
    for x in sbatch:
        pad = torch.zeros(lmax, dim)
        pad[: x.size(1), :] = x
        padded.append(pad.unsqueeze(0))
    X = torch.cat(padded, dim=0)
    return X, lens, lmax


def padding_samples(sbatch) -> Tuple[torch.Tensor, torch.Tensor, list, int]:
    lens = [x.size(1) for x in sbatch]
    lmax = max(lens)
    padded = []
    paddedmask = []
    for x in sbatch:
        pad = torch.zeros(lmax)
        pad[: x.size(1)] = x

        padded.append(pad.unsqueeze(0))
        # Masking
        mask = torch.BoolTensor(lmax).fill_(False)
        mask[x.size(1) :].fill_(True)
        paddedmask.append(mask.unsqueeze(0))

    X = torch.cat(padded, dim=0)
    Mask = torch.cat(paddedmask, dim=0)
    return X, Mask, lens, lmax


# Custom Dataset implementation for handling PeoplesSpeech data
class Custom_Speech_Dataset(TorchDatasetObj):
    def __init__(
        self,
        data_name="PS",
        dataset=None,
        features_type="LFB",
        txt_tokenizer=None,
        txt_model=None,
        acoustic_dim=80,
        sample_rate=8000,
        max_sentence_lenght=512,
    ) -> None:

        super(Custom_Speech_Dataset, self).__init__()

        # data
        self.dataset_name = data_name
        self.data = dataset

        # Acoustic features
        self.encoder_type = features_type
        self.sample_rate = sample_rate
        self.acoustic_dim = acoustic_dim

        # Text model
        self.bert_tokenizer = txt_tokenizer
        self.bert_model = txt_model
        # Tokenizer and maximum lenght of sentences
        self.max_sentence_length = max_sentence_lenght

        # These are to extract the 80-dimesional log-mel filterbank features
        self.compute_stft = STFT(
            sample_rate=self.sample_rate, win_length=25, hop_length=10, n_fft=400
        )
        self.compute_fbanks = Filterbank(n_mels=self.acoustic_dim)
        self.sr = sample_rate

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        # Processing audio
        audio_file = self.data[idx]["audio_path"]

        waveform = None
        sample_rate = None
        if self.dataset_name == "SLURP":
            waveform, sample_rate = torchaudio.load(audio_file)
        elif self.dataset_name == "PS" and self.encoder_type == "WCN":
            # We do not need acoustic file here, WCN have been precomputed
            wcn = self.data[idx]["wcn"]
        elif self.dataset_name == "PS" and self.encoder_type != "WCN":
            # For people speech  and NOT WCN
            waveform, sample_rate = torchaudio.load(audio_file)

        if self.encoder_type == "LFB":
            if sample_rate > self.sr:
                # subsampling is done only if LFB features are requested
                waveform = AT.Resample(sample_rate, self.sr)(waveform)
            acou_feats = self.get_filterbanks(waveform)
        elif self.encoder_type == "WCN":
            in_seqs, pos_seqs, score_seqs = (
                wcn["in_seq"],
                wcn["pos_seqs"],
                wcn["score_seqs"],
            )
            cls_tok = True
            # Vector of the position CLS is considered at the beggining of the sequence
            batch_pos = (
                [1] * cls_tok
                + [p + int(cls_tok) for p in pos_seqs]
                + [0] * (self.max_sentence_length - len(pos_seqs) - 1)
            )
            batch_pos = np.array(batch_pos)
            batch_score = (
                [1] * cls_tok
                + score_seqs
                + [-1] * (self.max_sentence_length - len(score_seqs) - 1)
            )
            batch_score = np.array(batch_score)

            # NOTE: when the encoding type is WCN, the acoustic features takes a different meaning
            # acou_feats will be an array of three elements as follows: [in_seqs,batch_pos,batch_score]
            acou_feats = [in_seqs, batch_pos, batch_score]

        # Cleaning the textual part and retuning
        utt = clean_str(self.data[idx]["ground_truth"])
        item = {"speech_feats": acou_feats, "utterance": utt}
        return item

    def get_filterbanks(self, signal) -> torch.Tensor:
        features = self.compute_stft(signal)
        features = spectral_magnitude(features)
        features = self.compute_fbanks(features)
        return features

    def build_padding_mask(self, x, x_len):
        padding_mask = torch.BoolTensor(x.shape[:2]).fill_(False)
        for i, l in enumerate(x_len):
            padding_mask[i, l:].fill_(True)
        return padding_mask


"""
General purpose class:
Identifies and loads the correct dataset depending on the task, pretrain or fine-tuning
"""


class WCN2B_Dataset(object):
    """
    Args:
       train: path to the training data
       test: path to the test data
       isDPP: Boleean variable that indicates running in distributed mode

    """

    # Dataset_root_path=''
    def __init__(
        self,
        dataset_name="PS",
        test_dataset_name="PS",
        train=None,
        dev=None,
        test=None,
        test_audios=None,
        isDPP=False,
        txt_model=None,
        features_type="LFB",
        acou_dimension=80,
        rank=0,
        world_size=0,
    ) -> None:

        self.dataset_name = dataset_name
        self.test_dataset_name = test_dataset_name
        self.train_path = train
        self.dev_path = dev
        self.test_path = test
        self.test_audios = test_audios

        self.isDPP = isDPP
        self.bert_tokenizer = AutoTokenizer.from_pretrained(txt_model)
        self.bert_model = AutoModel.from_pretrained(txt_model)

        # Pretrained Models
        self.encoder_type = features_type
        if self.encoder_type == "WCN":
            self.load_WCNs = True
            # Adding the following tokens are necessary for WCN representation
            self.bert_tokenizer.add_tokens(["<eps>", "<unk>"], special_tokens=True)
            self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))

        self.audio = "wav.scp"
        self.groundtruth = "text"

        self.train_set = list
        self.dev_set = list
        self.test_set = list
        self.samplerTrain = None
        self.samplerDev = None
        self.samplerTest = None
        self.max_sentence_length = 0
        self.pyr_layer = 3
        self.acou_dimension = acou_dimension
        self.rank = rank
        self.world_size = world_size

        print("Current working directory:", os.getcwd())

        if not os.path.exists(self.train_path):
            print("Folder does not exist.")
        elif not os.path.isdir(self.train_path):
            print("Path exists but is not a folder.")
        elif not os.access(self.train_path, os.R_OK):
            print("No read access to the folder.")
        else:
            print("Folder detected and accessible.")

        assert os.path.isdir(self.train_path), f"Error in the path: {self.train_path}"
        assert os.path.isdir(self.dev_path), f"Error in the path: {self.dev_path}"

        if int(self.rank) == 0:
            logger.info(f"Training data will be loaded from path: {self.train_path}")
        if int(self.rank) == 0:
            logger.info(f"Development data will be loaded from path: {self.dev_path}")
        if int(self.rank) == 0:
            logger.info(f"Test data will be loaded from path: {self.test_path}")

        if self.dataset_name == "PS":
            self.__getPeoples_Speech_data()
        elif self.dataset_name == "SLURP":
            self.audio = ".flac"
            self.__get_SLURP_data()
        # Here we load the test dataset to be used for evaluation of the model
        if self.test_dataset_name == "SLURP":
            self.audio = ".flac"
            self.__get_SLURP_data(only_test=True)

        if int(self.rank) == 0:
            logger.info(
                "\nDataset size:\n\tTrain set: {} examples\n\tDev set: {}\n\tTest set: {} examples".format(
                    len(self.train_set), len(self.dev_set), len(self.test_set)
                )
            )
        if int(self.rank) == 0:
            (
                logger.info("\n-->Running in DPP mode<--\n")
                if self.isDPP
                else logger.info("-->Running in single GPU/CPU mode<--")
            )

    def __get_SLURP_data(self, only_test=False):

        # Methos defined to do the pretraining using SLURP data
        if not only_test:
            # LOADING TRAIN PARTITION
            for root, directories, file_names in os.walk(self.train_path):
                file_names = [fi for fi in file_names if fi.endswith(".hrc2")]
                if len(file_names) > 0:
                    bar = Bar(
                        "Loading {} dataset: ".format(os.path.basename(root)),
                        max=len(file_names),
                    )
                    for filename in file_names:
                        bar.next()
                        slurp_example = self.__read_hrc2_files(
                            os.path.join(self.train_path, filename),
                            audios_root_path=self.test_audios,
                        )
                        if slurp_example is not None:
                            self.train_set.append(slurp_example)
                    del bar

            # LOADING dev PARTITION
            for root, directories, file_names in os.walk(self.dev_path):
                file_names = [fi for fi in file_names if fi.endswith(".hrc2")]
                if len(file_names) > 0:
                    bar = Bar(
                        "Loading {} dataset: ".format(os.path.basename(root)),
                        max=len(file_names),
                    )
                    for filename in file_names:
                        bar.next()
                        slurp_example = self.__read_hrc2_files(
                            os.path.join(self.dev_path, filename),
                            audios_root_path=self.test_audios,
                        )
                        if slurp_example is not None:
                            self.dev_set.append(slurp_example)
                    del bar
        else:
            # LOADING test PARTITION
            for root, directories, file_names in os.walk(self.test_path):
                file_names = [fi for fi in file_names if fi.endswith(".hrc2")]
                if len(file_names) > 0:
                    bar = Bar(
                        "Loading {} dataset: ".format(os.path.basename(root)),
                        max=len(file_names),
                    )
                    for filename in file_names:
                        bar.next()
                        slurp_example = self.__read_hrc2_files(
                            os.path.join(self.test_path, filename),
                            audios_root_path=self.test_audios,
                        )
                        if slurp_example is not None:
                            self.test_set.append(slurp_example)
                    del bar

    """Method that reads all the necessary data from People'sSpeech dataset"""

    def __getPeoples_Speech_data(self):
        # Reading training files  (Kaldi format)
        df_train_audios = self.__read_ps_wav_file(
            os.path.join(self.train_path, self.audio)
        )
        df_train_text = self.__read_text_file(
            os.path.join(self.train_path, self.groundtruth)
        )
        assert len(df_train_audios) == len(
            df_train_text
        ), f"Size does not match in training data {len(len(df_train_audios))} != {len(df_train_text)}"
        # Reading WCN files, WCN were pre-computed
        dict_train_wcns = self.read_wcns(
            os.path.join(self.train_path, "PepSpeech_train_WCN.csv")
        )
        bar = Bar(
            "Loading {} dataset: ".format(os.path.basename(self.train_path)),
            max=len(df_train_audios),
        )
        # FIXME
        max_num_data = 1000
        counter = 0
        for (idxRow, s1), (_, s2) in zip(
            df_train_audios.iterrows(), df_train_text.iterrows()
        ):
            bar.next()
            ps_example = self.__import_wcn_data(
                s1.tolist(), s2.tolist(), dict_train_wcns
            )
            if ps_example is not None:
                self.train_set.append(ps_example)
            counter += 1
            if counter == max_num_data:
                break
        del bar
        # Freeing up memory
        del df_train_audios
        del df_train_text
        del dict_train_wcns

        # Loading DEV files
        df_dev_audios = self.__read_ps_wav_file(os.path.join(self.dev_path, self.audio))
        df_dev_text = self.__read_text_file(
            os.path.join(self.dev_path, self.groundtruth)
        )
        assert len(df_dev_audios) == len(
            df_dev_text
        ), f"Size does not match in training data {len(df_dev_audios)} != {len(df_dev_text)}"
        # Reading WCN files, WCN were pre-computed
        dict_dev_wcns = self.read_wcns(
            os.path.join(self.dev_path, "PepSpeech_test_WCN.csv")
        )
        bar = Bar(
            "Loading {} dataset: ".format(os.path.basename(self.dev_path)),
            max=len(df_dev_audios),
        )
        # FIXME
        counter = 0
        for (idxRow, s1), (_, s2) in zip(
            df_dev_audios.iterrows(), df_dev_text.iterrows()
        ):
            bar.next()
            ps_example = self.__import_wcn_data(s1.tolist(), s2.tolist(), dict_dev_wcns)
            if ps_example is not None:
                self.dev_set.append(ps_example)
            counter += 1
            if counter == max_num_data:
                break
        del bar
        # Freeing up memory
        del df_dev_audios
        del df_dev_text
        del dict_dev_wcns

    def get_vocabulary_size(self) -> int:
        return len(self.bert_tokenizer)

    def __my_collate_batch(self, batch):
        if self.encoder_type == "LFB":  # Means to get LFB-based features
            speech_batch = [
                pad(x["speech_feats"].squeeze(0), factor=2**self.pyr_layer).unsqueeze(0)
                for x in batch
                if x["speech_feats"].size(1) > 2
            ]
            X, lens, lmax = padding(speech_batch)
            speech_mask = get_mask(lens)
            # Not required for this encoding
            in_seqs_raw, pos_seqs_raw, score_seqs_raw = None, None, None
        elif (
            self.encoder_type == "WCN"
        ):  # Getting WCN values acou_feats=[in_seqs,batch_pos,batch_score]
            # These variables are not used for WCN encoding
            X, lens, lmax, speech_mask = None, None, None, None
            # Getting WCN data and generating tensors
            in_seqs_raw = [x["speech_feats"][0] for x in batch]
            pos_seqs = [x["speech_feats"][1] for x in batch]
            pos_seqs_raw = [torch.tensor(x).long() for x in pos_seqs]
            score_seqs = [x["speech_feats"][2] for x in batch]
            score_seqs_raw = [torch.tensor(x).long() for x in score_seqs]

        # Getting Textual features
        text_raw = [x["utterance"] for x in batch]
        text_batch = self.bert_tokenizer(
            text_raw, return_tensors="pt", padding=True, truncation=True
        )
        text_batch_unpad = self.bert_tokenizer(text_raw).input_ids
        text_batch_raw = [torch.tensor(x).long() for x in text_batch_unpad]

        items = {
            "speech_feats": X,
            "speech_lens": lens,
            "speech_max_len": lmax,
            "speech_mask": speech_mask,
            "textual_batch": text_batch,
            "text_batch_raw": text_batch_raw,
            "in_seqs": in_seqs_raw,
            "pos_seqs": pos_seqs_raw,
            "score_seqs": score_seqs_raw,
        }
        return items

    def generate_DataLoaderObject(self, bs=32, output_folder="output") -> DataLoader:
        # Creating folder where all the generated outputs will be saved
        if int(self.rank) == 0 and not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Generating the TRAINING Dataloader Object
        train_obj = Custom_Speech_Dataset(
            data_name=self.dataset_name,
            dataset=self.train_set,
            features_type=self.encoder_type,
            txt_tokenizer=self.bert_tokenizer,
            txt_model=self.bert_model,
            acoustic_dim=self.acou_dimension,
            max_sentence_lenght=self.max_sentence_length,
        )
        # If distributed learning is True
        if self.isDPP:
            self.samplerTrain = torch.utils.data.distributed.DistributedSampler(
                train_obj
            )
            assert isinstance(self.world_size, int) and self.world_size > 0
            batch_size = bs // self.world_size
            train_DL = DataLoader(
                train_obj,
                num_workers=0,
                collate_fn=self.__my_collate_batch,
                batch_size=batch_size,
                sampler=self.samplerTrain,
            )
        else:
            self.samplerTrain = None
            batch_size = bs
            train_DL = DataLoader(
                train_obj,
                num_workers=0,
                collate_fn=self.__my_collate_batch,
                batch_size=batch_size,
                shuffle=True,
            )

        # Generating the DEVELOPMENT Dataloader Object
        if int(self.rank) == 0:
            logger.info(f"\nBatch size used for train, dev and test {batch_size}")

        dev_obj = Custom_Speech_Dataset(
            data_name=self.dataset_name,
            dataset=self.dev_set,
            features_type=self.encoder_type,
            txt_tokenizer=self.bert_tokenizer,
            txt_model=self.bert_model,
            acoustic_dim=self.acou_dimension,
            max_sentence_lenght=self.max_sentence_length,
        )

        if self.isDPP:
            self.samplerDevel = torch.utils.data.distributed.DistributedSampler(dev_obj)
            assert isinstance(self.world_size, int) and self.world_size > 0
            batch_size = bs // self.world_size
            dev_DL = DataLoader(
                dev_obj,
                num_workers=0,
                collate_fn=self.__my_collate_batch,
                batch_size=batch_size,
                sampler=self.samplerDevel,
            )
        else:
            self.samplerDevel = None
            dev_DL = DataLoader(
                dev_obj,
                num_workers=0,
                collate_fn=self.__my_collate_batch,
                batch_size=batch_size,
                shuffle=False,
            )

        # Generating the Test Dataloader Object
        test_obj = Custom_Speech_Dataset(
            data_name=self.test_dataset_name,
            dataset=self.test_set,
            features_type=self.encoder_type,
            txt_tokenizer=self.bert_tokenizer,
            txt_model=self.bert_model,
            acoustic_dim=self.acou_dimension,
            max_sentence_lenght=self.max_sentence_length,
        )
        self.samplerTest = None
        test_DL = DataLoader(
            test_obj,
            num_workers=0,
            collate_fn=self.__my_collate_batch,
            batch_size=batch_size,
            shuffle=False,
        )
        # Returning Dataloader Objects
        return train_DL, dev_DL, test_DL

    """Method for reading the WAV.SCP files for People's Speech"""

    def __read_ps_wav_file(self, fname) -> pd.DataFrame:
        ids = []
        audios = []
        wav_file = open(fname, "r")
        wav_lines = wav_file.readlines()
        for line in wav_lines:
            columns = line.split(" ")
            ids.append(columns[0])
            path = columns[3]
            audios.append(path)
        return pd.DataFrame(list(zip(ids, audios)), columns=["id", "audio_path"])

    """Method for reading the TEXT files"""

    def __read_text_file(self, fname) -> pd.DataFrame:
        ids = []
        text = []
        text_file = open(fname, "r")
        text_lines = text_file.readlines()
        for line in text_lines:
            line = line.strip()
            columns = line.split(" ", 1)
            ids.append(columns[0])
            text.append(columns[1])
        return pd.DataFrame(list(zip(ids, text)), columns=["id", "text"])

    """Method for reading the precomputed word_confussion_networks"""

    def read_wcns(self, file_path):
        wcns_dict = dict()
        with open(file_path, "r") as reader:
            # Read and print the entire file line by line
            line = reader.readline()
            while line != "":  # The EOF char is an empty string
                wcn_id, wcn_string = line.split(",")
                wcn_string = wcn_string.strip("\n\r")
                wcn_string = wcn_string.replace("['", "")
                wcn_string = wcn_string.replace("']", "")
                wcn_string = wcn_string.replace('["', "")
                wcn_string = wcn_string.replace('"]', "")
                wcn_string = wcn_string.strip()
                if wcn_id not in wcns_dict.keys():
                    wcns_dict[wcn_id] = wcn_string
                else:
                    logger.info(f"Id is repeated for entry: {wcn_id}")
                line = reader.readline()
        return wcns_dict

    """Reading the SLURP files"""

    def __read_hrc2_files(self, input_file, audios_root_path=None) -> dict:
        try:
            huric_example_xml = et.parse(input_file)
        except et.ParseError:
            logger.info("Problems reading file: {}".format(input_file))
            return None
        root = huric_example_xml.getroot()
        example_id = root.attrib["id"]
        example = dict()
        example["id"] = example_id
        audio_id = root.attrib["audio_id"]
        audio_id = os.path.join(audios_root_path, audio_id + self.audio)
        transcription = ""
        for sentence in root.findall("sentence"):
            transcription = sentence.text.encode("utf-8")
        example["audio_path"] = audio_id
        example["ground_truth"] = str(transcription.decode("utf8"))
        sentence_length = len(str(transcription).split(" "))
        if sentence_length > self.max_sentence_length:
            self.max_sentence_length = sentence_length
        example["sentence_length"] = sentence_length
        return example

    """Method for importing the WCN files and data"""

    def __import_wcn_data(self, audio_inf=None, text_inf=None, wcns=None):
        """
        * fn: wcn data file name
        * line format - word:parent:sibling:type ... \t<=>\tword:pos:score word:pos:score ... \t<=>\tlabel1;label2...
        * system act <=> utterance <=> labels
        """
        assert (
            audio_inf[0] == text_inf[0]
        ), f"Ids of the input rows do not match,\
             this sample will not be included: {audio_inf[0]}"

        peoplesSpeech_example = dict()
        peoplesSpeech_example["id"] = audio_inf[0]
        peoplesSpeech_example["audio_path"] = audio_inf[1]
        peoplesSpeech_example["ground_truth"] = str(text_inf[1])

        # lists to store the WCN information
        in_seqs = []
        pos_seqs = []
        score_seqs = []
        try:
            wcn_info = wcns[audio_inf[0]]
            wcn_info = wcn_info.strip()
        except Exception:
            logger.info(f"AudioID {audio_inf[0]} does not have a WCN data")
            return None

        inp_list = wcn_info.split(" ")

        in_seq, pos_seq, score_seq = zip(
            *[item.strip().split(":") for item in inp_list]
        )
        in_seqs = list(in_seq)
        pos_seqs = list(map(int, pos_seq))
        score_seqs = list(map(float, score_seq))
        wcn_dict = dict()
        wcn_dict["in_seq"] = in_seqs
        wcn_dict["pos_seqs"] = pos_seqs
        wcn_dict["score_seqs"] = score_seqs
        peoplesSpeech_example["wcn"] = wcn_dict
        # Ground truth sentence lenght
        sentence_length = self.tokenize_wcn(in_seqs)
        if sentence_length > self.max_sentence_length:
            self.max_sentence_length = sentence_length
        peoplesSpeech_example["sentence_length"] = sentence_length
        return peoplesSpeech_example

    """Method to measure the lenght of the tokenized elements of the WCN, necessary for bacthing"""

    def tokenize_wcn(self, sentence=None):
        tokenized_sentence = []
        for word in sentence:
            # Tokenize the word
            tokenized_word = self.bert_tokenizer.tokenize(word)
            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)
        return len(tokenized_sentence)

    def get_train_sampler(self):
        return self.samplerTrain

    def get_dev_sampler(self):
        return self.samplerDev

    def get_test_sampler(self):
        return self.samplerTest


"""Custom Dataset implementation for handling SLURP data"""


class Custom_SLURP_Dataset(TorchDatasetObj):
    def __init__(
        self,
        dataset=None,
        label="intent",
        features_type="LFB",
        txt_tokenizer=None,
        txt_model=None,
        acoustic_dim=80,
        sample_rate=8000,
        max_sentence_lenght=512,
        label_encoder=None,
    ) -> None:
        super(Custom_SLURP_Dataset, self).__init__()

        # data and classification task
        self.data = dataset
        self.classification_task = label

        # Acoustic features
        self.encoder_type = features_type
        self.sr = sample_rate
        self.acoustic_dim = acoustic_dim

        # Text model
        self.bert_tokenizer = txt_tokenizer
        self.bert_model = txt_model

        # Tokenizer and maximum lenght of sentences
        self.max_sentence_length = max_sentence_lenght

        # These are to extract the X-dimesional log-mel filterbank features
        self.compute_stft = STFT(
            sample_rate=8000, win_length=25, hop_length=10, n_fft=400
        )
        self.compute_fbanks = Filterbank(n_mels=self.acoustic_dim)

        self.label_encoder = label_encoder

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        # Processing audio
        audio_file = self.data[idx]["audio_path"]
        waveform, sample_rate = None, None
        if self.encoder_type == "LFB":
            waveform, sample_rate = torchaudio.load(audio_file)
            if sample_rate > self.sr:
                waveform = AT.Resample(sample_rate, self.sr)(waveform)
            acou_feats = self.get_filterbanks(waveform)
        elif self.encoder_type == "WCN":
            wcn = self.data[idx]["wcn"]
            in_seqs, pos_seqs, score_seqs = (
                wcn["in_seq"],
                wcn["pos_seqs"],
                wcn["score_seqs"],
            )
            cls_tok = True
            # Vector of the positions CLS is considered at the beggining of the sequence
            batch_pos = (
                [1] * cls_tok
                + [p + int(cls_tok) for p in pos_seqs]
                + [0] * (self.max_sentence_length - len(pos_seqs) - 1)
            )
            batch_pos = np.array(batch_pos)
            # Vector of the scores CLS is considered at the beggining of the sequence
            batch_score = (
                [1] * cls_tok
                + score_seqs
                + [-1] * (self.max_sentence_length - len(score_seqs) - 1)
            )
            batch_score = np.array(batch_score)
            # NOTE: when encoding type is WCN, acoustic features take a different meaning
            # acou_feats will be an array of three elements as follows: [in_seqs,batch_pos,batch_score]
            acou_feats = [in_seqs, batch_pos, batch_score]

        # Cleaning the textual part and retuning
        utt = clean_str(self.data[idx]["ground_truth"])
        # Retrieving and encoding label
        label = self.label_encoder.transform(
            [self.data[idx][self.classification_task]]
        )[0]
        item = {"speech_feats": acou_feats, "utterance": utt, "label": label}
        return item

    def get_filterbanks(self, signal) -> torch.Tensor:
        features = self.compute_stft(signal)
        features = spectral_magnitude(features)
        features = self.compute_fbanks(features)
        return features


class SLURP_Dataset(object):
    def __init__(
        self,
        train=None,
        dev=None,
        test=None,
        test_audios=None,
        isDPP=False,
        txt_model=None,
        acou_model=None,
        features_type=None,
        acou_dimension=80,
        output_folder="\\",
        clasification_task="intent",
        rank=0,
        world_size=0,
        train_wcn=None,
        dev_wcn=None,
        test_wcn=None,
    ) -> None:

        self.train_path = train
        self.dev_path = dev
        self.test_path = test
        self.test_audios = test_audios

        self.isDPP = isDPP
        self.load_WCNs = False
        self.skkiped_files = []
        self.bert_tokenizer = AutoTokenizer.from_pretrained(txt_model)
        self.bert_model = AutoModel.from_pretrained(txt_model)
        self.encoder_type = features_type
        self.acou_dimension = acou_dimension
        if self.encoder_type == "WCN":
            self.load_WCNs = True
            self.files_without_wcn = 0
            self.bert_tokenizer.add_tokens(["<eps>", "<unk>"], special_tokens=True)
            self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))

        self.audio = ".flac"
        self.groundtruth = "text"

        self.train_set = []
        self.dev_set = []
        self.test_set = []
        self.samplerTrain = None
        self.samplerDevel = None
        self.pyr_layer = 3
        self.max_sentence_length = 0
        self.label_encoder = None
        self.logging_folder = output_folder
        self.cls_task = clasification_task
        self.rank = rank
        self.world_size = world_size

        assert os.path.isdir(self.train_path), f"Error in the path: {self.train_path}"
        assert os.path.isdir(self.dev_path), f"Error in the path: {self.dev_path}"
        assert os.path.isdir(self.test_path), f"Error in the path: {self.test_path}"

        if int(rank) == 0:
            logger.info(
                "Training data will be loaded from path: `{}`".format(self.train_path)
            )
        if int(rank) == 0:
            logger.info(
                "Development data will be loaded from path: `{}`".format(self.dev_path)
            )
        if int(rank) == 0:
            logger.info(
                "Test data will be loaded from path: `{}`".format(self.test_path)
            )

        # Loading WCN from SLURP (ONLY IF ACTIVATED THE WCN encoding type)
        if self.load_WCNs:
            logger.info("Loading Word Consensus Networks for SLURP data")
            self.train_WCN_dict = self.__read_wcn_data(train_wcn)
            self.dev_WCN_dict = self.__read_wcn_data(dev_wcn)
            self.test_WCN_dict = self.__read_wcn_data(test_wcn)
        else:
            self.train_WCN_dict = None
            self.dev_WCN_dict = None
            self.test_WCN_dict = None
        # LOADING TRAIN PARTITION
        for root, directories, file_names in os.walk(self.train_path):
            file_names = [fi for fi in file_names if fi.endswith(".hrc2")]
            if len(file_names) > 0:
                bar = Bar(
                    "Loading {} dataset: ".format(os.path.basename(root)),
                    max=len(file_names),
                )
                for filename in file_names:
                    bar.next()
                    slurp_example = self.__read_hrc2_files(
                        os.path.join(self.train_path, filename),
                        partition="train",
                        audios_root_path=self.test_audios,
                    )
                    if slurp_example is not None:
                        self.train_set.append(slurp_example)
                del bar
        # LOADING dev PARTITION
        for root, directories, file_names in os.walk(self.dev_path):
            file_names = [fi for fi in file_names if fi.endswith(".hrc2")]
            if len(file_names) > 0:
                bar = Bar(
                    "Loading {} dataset: ".format(os.path.basename(root)),
                    max=len(file_names),
                )
                for filename in file_names:
                    bar.next()
                    slurp_example = self.__read_hrc2_files(
                        os.path.join(self.dev_path, filename),
                        partition="dev",
                        audios_root_path=self.test_audios,
                    )
                    if slurp_example is not None:
                        self.dev_set.append(slurp_example)
                del bar
        # LOADING test PARTITION
        for root, directories, file_names in os.walk(self.test_path):
            file_names = [fi for fi in file_names if fi.endswith(".hrc2")]
            if len(file_names) > 0:
                bar = Bar(
                    "Loading {} dataset: ".format(os.path.basename(root)),
                    max=len(file_names),
                )
                for filename in file_names:
                    bar.next()
                    slurp_example = self.__read_hrc2_files(
                        os.path.join(self.test_path, filename),
                        partition="test",
                        audios_root_path=self.test_audios,
                    )
                    if slurp_example is not None:
                        self.test_set.append(slurp_example)
                del bar
        if int(rank) == 0:
            logger.info(f"Total number of files skkiped {len(self.skkiped_files)}")
        # Generating label encoders for intent
        # If you want to predict dialogue act need to manually modify the label parameter
        self.label_encoder = self.__generate_label_encoders_for_single_out(
            label=self.cls_task, save_encoders=True, rank=rank
        )
        if int(rank) == 0:
            logger.info(
                "\nDataset size:\n\tTrain set: {} examples\n\tDev set: {} examples \n\tTest set: {} examples\n".format(
                    len(self.train_set), len(self.dev_set), len(self.test_set)
                )
            )
        if int(rank) == 0:
            logger.info(
                f"Total number of '{self.cls_task}' classes: {self.num_labels}\n"
            )
        if int(rank) == 0:
            (
                logger.info("-->Running in DPP mode<--")
                if self.isDPP
                else logger.info("-->Running in single GPU/CPU mode<--")
            )
        if self.load_WCNs:
            if int(rank) == 0:
                logger.info("WCNs loaded sucessfully!!")

    def get_vocabulary_size(self) -> int:
        return len(self.bert_tokenizer)

    def get_num_classes(self) -> int:
        return self.num_labels

    """Getting the WCN input """

    def __get_WCN(self, id, partition="train"):
        try:
            if partition == "train":
                return self.train_WCN_dict[id]
            elif partition == "dev":
                return self.dev_WCN_dict[id]
            elif partition == "test":
                return self.test_WCN_dict[id]
        except KeyError:
            logger.info(f"\nNo WCN data for:{id}")
            self.files_without_wcn += 1
            return None

    """Method for reading the WCN files from SLURP"""

    def __read_wcn_data(self, fn):
        """
        * fn: wcn data file name
        * line format - word:parent:sibling:type ... \t<=>\tword:pos:score word:pos:score ... \t<=>\tlabel1;label2...
        * system act <=> utterance <=> labels
        """
        wcn_dict = {}
        in_seqs = []
        pos_seqs = []
        score_seqs = []
        sa_seqs = []
        sa_parent_seqs = []
        sa_sib_seqs = []
        sa_type_seqs = []
        labels = []
        with open(fn, "r") as fp:
            lines = fp.readlines()
            for line in lines:
                id, sa, inp, lbl = line.strip("\n\r").split("\t<=>\t")
                inp_list = inp.strip().split(" ")
                in_seq, pos_seq, score_seq = zip(
                    *[item.strip().split(":") for item in inp_list]
                )
                in_seqs = list(in_seq)
                pos_seqs = list(map(int, pos_seq))
                score_seqs = list(map(float, score_seq))
                sa_list = sa.strip().split(" ")
                sa_seq, pa_seq, sib_seq, ty_seq = zip(
                    *[item.strip().split(":") for item in sa_list]
                )
                sa_seqs = list(sa_seq)
                sa_parent_seqs = list(map(int, pa_seq))
                sa_sib_seqs = list(map(int, sib_seq))
                sa_type_seqs = list(map(int, ty_seq))

                if len(lbl) == 0:
                    labels = []
                else:
                    labels = lbl.strip().split(";")
                if id not in wcn_dict.keys():
                    wcn_dict[id] = [
                        in_seqs,
                        pos_seqs,
                        score_seqs,
                        sa_seqs,
                        sa_parent_seqs,
                        sa_sib_seqs,
                        sa_type_seqs,
                        labels,
                    ]
        return wcn_dict

    """Method to measure the lenght of the tokenized elements of the WCN, necessary for bacthing"""

    def tokenize_wcn(self, sentence=None):
        tokenized_sentence = []
        for word in sentence:
            # Tokenize the word
            tokenized_word = self.bert_tokenizer.tokenize(word)
            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)
        return len(tokenized_sentence)

    def __generate_label_encoders_for_single_out(
        self, label="intent", save_encoders=False, rank=0
    ) -> preprocessing.LabelEncoder:
        bag_of_labels = set()

        for example in self.train_set:
            bag_of_labels.add(example[label])

        for example in self.dev_set:
            bag_of_labels.add(example[label])

        for example in self.test_set:
            bag_of_labels.add(example[label])

        bag_of_labels = list(bag_of_labels)
        # Getting the number of classes
        self.num_labels = len(bag_of_labels)

        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(bag_of_labels)

        f_out = os.path.join(self.logging_folder, "labels_encoder")
        if save_encoders:
            if int(rank) == 0 and not os.path.exists(f_out):
                os.makedirs(f_out)
            np.save(os.path.join(f_out, label + "_labels.npy"), label_encoder.classes_)
        return label_encoder

    def __read_hrc2_files(
        self, input_file, partition="train", audios_root_path=None
    ) -> dict:
        try:
            huric_example_xml = et.parse(input_file)
        except et.ParseError:
            logger.info("Problems reading file: {}".format(input_file))
            return None
        root = huric_example_xml.getroot()
        example_id = root.attrib["id"]
        example = dict()
        example["id"] = example_id
        audio_id = root.attrib["audio_id"]
        audio_id = os.path.join(audios_root_path, audio_id + self.audio)
        transcription = ""
        for sentence in root.findall("sentence"):
            transcription = sentence.text.encode("utf-8")
        example["audio_path"] = audio_id
        example["ground_truth"] = str(transcription.decode("utf8"))
        sentence_length = len(str(transcription).split(" "))
        if sentence_length > self.max_sentence_length:
            self.max_sentence_length = sentence_length
        example["sentence_length"] = sentence_length
        # getting labels in sequence form
        try:
            dialogue_act_annotations = np.full(
                sentence_length, fill_value="O", dtype="object"
            )
            frame_annotations = np.full(sentence_length, fill_value="O", dtype="object")
            for dialogue_act in root.findall("./semantics/dialogueAct/token"):
                dialogue_act_annotations[int(dialogue_act.attrib["id"]) - 1] = (
                    dialogue_act.attrib["value"]
                )
            for frame in root.findall("./semantics/frame/token"):
                frame_annotations[int(frame.attrib["id"]) - 1] = frame.attrib["value"]
            # Instead of sequence labelling, we only need a single label for the entire utterance
            example["dialogue_act"] = self.__get_single_label(dialogue_act_annotations)
            example["intent"] = self.__get_single_label(frame_annotations)
            # Getting WCN' information
            if self.load_WCNs:
                wcn_raw_data = self.__get_WCN(os.path.basename(input_file), partition)
                if wcn_raw_data is None:
                    raise IndexError()
                else:
                    wcn_dict = dict()
                    wcn_dict["in_seq"] = wcn_raw_data[0]
                    wcn_dict["pos_seqs"] = wcn_raw_data[1]
                    wcn_dict["score_seqs"] = wcn_raw_data[2]
                    example["wcn"] = wcn_dict
                    # Updating lenghts based on the WCN info
                    tokenized_sentence_lenght = self.tokenize_wcn(wcn_dict["in_seq"])
                    if tokenized_sentence_lenght > self.max_sentence_length:
                        self.max_sentence_length = tokenized_sentence_lenght
                    example["sentence_length"] = tokenized_sentence_lenght
            return example
        except IndexError:
            self.skkiped_files.append(input_file)
            return None

    def __get_single_label(self, annotations) -> str:
        assert len(annotations) >= 0
        label = annotations[0]
        label = label.split("-")[1]
        return str(label)

    def __my_collate_batch(self, batch):
        in_seqs_raw = None
        pos_seqs = None
        pos_seqs_raw = None
        score_seqs = None
        score_seqs_raw = None
        if self.encoder_type == "LFB":
            # Getting LFB-based features
            speech_batch = [
                pad(x["speech_feats"].squeeze(0), factor=2**self.pyr_layer).unsqueeze(0)
                for x in batch
                if x["speech_feats"].size(1) > 2
            ]
            X, lens, lmax = padding(speech_batch)
            speech_mask = get_mask(lens)
        elif (
            self.encoder_type == "WCN"
        ):  # Getting WCN values acou_feats=[in_seqs,batch_pos,batch_score]
            # These variables are not used for WCN encoding
            X, lens, lmax, speech_mask = None, None, None, None
            # Getting WCN data and generating tensors
            in_seqs_raw = [x["speech_feats"][0] for x in batch]
            pos_seqs = [x["speech_feats"][1] for x in batch]
            pos_seqs_raw = [torch.tensor(x).long() for x in pos_seqs]
            score_seqs = [x["speech_feats"][2] for x in batch]
            score_seqs_raw = [torch.tensor(x).long() for x in score_seqs]

        # Getting Textual features
        text_raw = [x["utterance"] for x in batch]
        text_batch = self.bert_tokenizer(
            text_raw, return_tensors="pt", padding=True, truncation=True
        )
        text_batch_unpad = self.bert_tokenizer(text_raw).input_ids
        text_batch_raw = [torch.tensor(x).long() for x in text_batch_unpad]
        # Getting labels
        label_raw = [x["label"] for x in batch]
        label_batch = torch.tensor(label_raw).long()

        items = {
            "speech_feats": X,
            "speech_lens": lens,
            "speech_max_len": lmax,
            "speech_mask": speech_mask,
            "textual_batch": text_batch,
            "text_batch_raw": text_batch_raw,
            "in_seqs": in_seqs_raw,
            "pos_seqs": pos_seqs_raw,
            "score_seqs": score_seqs_raw,
            "labels_batch_raw": label_batch,
        }
        return items

    def generate_DataLoaderObject(self, bs=32) -> DataLoader:
        # Creating folder where all the generated outouts will be saved
        if self.rank == 0 and not os.path.exists(self.logging_folder):
            os.makedirs(self.logging_folder)

        # Generating the TRAINING Dataloader Object
        train_obj = Custom_SLURP_Dataset(
            dataset=self.train_set,
            features_type=self.encoder_type,
            txt_tokenizer=self.bert_tokenizer,
            txt_model=self.bert_model,
            acoustic_dim=self.acou_dimension,
            max_sentence_lenght=self.max_sentence_length,
            label_encoder=self.label_encoder,
        )

        if self.isDPP:
            self.samplerTrain = torch.utils.data.distributed.DistributedSampler(
                train_obj
            )
            world_size = torch.distributed.get_world_size()
            assert isinstance(world_size, int) and world_size > 0
            batch_size = bs // world_size
            train_DL = DataLoader(
                train_obj,
                collate_fn=self.__my_collate_batch,
                batch_size=batch_size,
                sampler=self.samplerTrain,
                num_workers=0,
            )
        else:
            self.samplerTrain = None
            batch_size = bs
            train_DL = DataLoader(
                train_obj,
                collate_fn=self.__my_collate_batch,
                batch_size=batch_size,
                shuffle=True,
            )

        # Generating the DEVELOPMENT Dataloader Object
        dev_obj = Custom_SLURP_Dataset(
            dataset=self.dev_set,
            features_type=self.encoder_type,
            txt_tokenizer=self.bert_tokenizer,
            txt_model=self.bert_model,
            acoustic_dim=self.acou_dimension,
            max_sentence_lenght=self.max_sentence_length,
            label_encoder=self.label_encoder,
        )

        if self.isDPP:
            self.samplerDevel = torch.utils.data.distributed.DistributedSampler(dev_obj)
            assert isinstance(self.world_size, int) and self.world_size > 0
            batch_size = bs // self.world_size
            dev_DL = DataLoader(
                dev_obj,
                num_workers=0,
                collate_fn=self.__my_collate_batch,
                batch_size=batch_size,
                sampler=self.samplerDevel,
            )
        else:
            self.samplerDevel = None
            dev_DL = DataLoader(
                dev_obj,
                num_workers=0,
                collate_fn=self.__my_collate_batch,
                batch_size=batch_size,
                shuffle=False,
            )

        # Generating the Test Dataloader Object
        test_obj = Custom_SLURP_Dataset(
            dataset=self.test_set,
            features_type=self.encoder_type,
            txt_tokenizer=self.bert_tokenizer,
            txt_model=self.bert_model,
            acoustic_dim=self.acou_dimension,
            max_sentence_lenght=self.max_sentence_length,
            label_encoder=self.label_encoder,
        )

        test_DL = DataLoader(
            test_obj,
            collate_fn=self.__my_collate_batch,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        return train_DL, dev_DL, test_DL

    def get_train_sampler(self):
        return self.samplerTrain
