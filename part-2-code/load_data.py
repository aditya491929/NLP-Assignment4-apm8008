import os
import random
import re
import string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download("punkt", quiet=True)

from transformers import T5TokenizerFast
import torch

PAD_IDX = 0


class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        """
        Dataset class for the T5 model.

        Behavior:
          * Uses 'google-t5/t5-small' tokenizer for both encoder and decoder.
          * For train/dev: returns encoder ids, decoder inputs, decoder targets,
            and the initial decoder input token.
          * For test: returns only encoder ids and initial decoder input token.

        Args:
            data_folder (str): path to data directory
            split (str): "train", "dev", or "test"
        """
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.max_source_len = 256
        self.max_target_len = 256
        self.bos_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        self.eos_id = self.tokenizer.eos_token_id

        self.examples = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        examples = []

        if split in ("train", "dev"):
            nl_path = os.path.join(data_folder, f"{split}.nl")
            sql_path = os.path.join(data_folder, f"{split}.sql")

            with open(nl_path, "r") as f:
                nl_lines = [l.strip() for l in f.readlines()]
            with open(sql_path, "r") as f:
                sql_lines = [l.strip() for l in f.readlines()]

            assert len(nl_lines) == len(
                sql_lines
            ), "Mismatched NL and SQL line counts."

            for nl, sql in zip(nl_lines, sql_lines):
                # Encoder: NL
                enc = tokenizer(
                    nl,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_source_len,
                )
                encoder_ids = torch.tensor(
                    enc["input_ids"], dtype=torch.long
                )

                # Decoder: SQL
                sql_enc = tokenizer(
                    sql,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_target_len - 2,  # keep room for BOS+EOS
                )
                sql_ids = sql_enc["input_ids"]

                decoder_input_ids = torch.tensor(
                    [self.bos_id] + sql_ids, dtype=torch.long
                )
                decoder_target_ids = torch.tensor(
                    sql_ids + [self.eos_id], dtype=torch.long
                )
                initial_decoder_input = torch.tensor(
                    [self.bos_id], dtype=torch.long
                )

                examples.append(
                    {
                        "encoder_ids": encoder_ids,
                        "decoder_inputs": decoder_input_ids,
                        "decoder_targets": decoder_target_ids,
                        "initial_decoder_input": initial_decoder_input,
                        "raw_nl": nl,
                        "raw_sql": sql,
                    }
                )

        elif split == "test":
            nl_path = os.path.join(data_folder, "test.nl")
            with open(nl_path, "r") as f:
                nl_lines = [l.strip() for l in f.readlines()]

            for nl in nl_lines:
                enc = tokenizer(
                    nl,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_source_len,
                )
                encoder_ids = torch.tensor(
                    enc["input_ids"], dtype=torch.long
                )
                initial_decoder_input = torch.tensor(
                    [self.bos_id], dtype=torch.long
                )

                examples.append(
                    {
                        "encoder_ids": encoder_ids,
                        "initial_decoder_input": initial_decoder_input,
                        "raw_nl": nl,
                    }
                )
        else:
            raise ValueError(f"Unknown split: {split}")

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def normal_collate_fn(batch):
    """
    Collation function to perform dynamic padding for training and evaluation
    with the development set.

    Returns:
        encoder_ids (B, T)
        encoder_mask (B, T)
        decoder_inputs (B, T')
        decoder_targets (B, T')
        initial_decoder_inputs (B, 1)
    """
    encoder_list = [b["encoder_ids"] for b in batch]
    decoder_in_list = [b["decoder_inputs"] for b in batch]
    decoder_tgt_list = [b["decoder_targets"] for b in batch]
    init_dec_list = [b["initial_decoder_input"] for b in batch]

    encoder_ids = pad_sequence(
        encoder_list, batch_first=True, padding_value=PAD_IDX
    )
    decoder_inputs = pad_sequence(
        decoder_in_list, batch_first=True, padding_value=PAD_IDX
    )
    decoder_targets = pad_sequence(
        decoder_tgt_list, batch_first=True, padding_value=PAD_IDX
    )
    initial_decoder_inputs = pad_sequence(
        init_dec_list, batch_first=True, padding_value=PAD_IDX
    )

    encoder_mask = (encoder_ids != PAD_IDX).long()

    return (
        encoder_ids,
        encoder_mask,
        decoder_inputs,
        decoder_targets,
        initial_decoder_inputs,
    )


def test_collate_fn(batch):
    """
    Collation for test inference.

    Returns:
        encoder_ids (B, T)
        encoder_mask (B, T)
        initial_decoder_inputs (B, 1)
    """
    encoder_list = [b["encoder_ids"] for b in batch]
    init_dec_list = [b["initial_decoder_input"] for b in batch]

    encoder_ids = pad_sequence(
        encoder_list, batch_first=True, padding_value=PAD_IDX
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = pad_sequence(
        init_dec_list, batch_first=True, padding_value=PAD_IDX
    )

    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = "data"
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_prompting_data(data_folder):
    """
    Simple helper for the prompting part (if needed).
    Uses the same files: train.nl/sql, dev.nl/sql, test.nl.

    Returns:
        train_x, train_y, dev_x, dev_y, test_x
    """
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x