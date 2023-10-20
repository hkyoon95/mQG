from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import pytorch_lightning as pl
import os

cur_dir = os.path.dirname(os.path.realpath(__file__))


class FTQADataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_token_len
    ):
        super().__init__()
        self.max_source_length = max_token_len
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.df = data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """Call tokenizer on src and tgt_lines"""
        source_line = self.df.loc[index, 'cor_section']
        q_line = self.df.loc[index, 'question']
        start_position = torch.LongTensor([self.df.loc[index, 'start_token']])
        end_postion = torch.LongTensor([self.df.loc[index, 'end_token']])
        source_inputs = self.tokenizer.encode_plus(['<IMPLICIT>'+q_line, source_line], max_length=self.max_source_length, 
                                    padding="max_length", return_tensors='pt', truncation=True)
        source_ids = source_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "start_positions": start_position,
            "end_positions": end_postion
        }

    def collate_fn(self, batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        start_positions = torch.stack([x["start_positions"] for x in batch])
        end_positions = torch.stack([x["end_positions"] for x in batch])
        pad_token_id = self.pad_token_id

        src_ids, src_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": src_ids,
            "attention_mask": src_mask,
            "start_positions": start_positions,
            "end_positions": end_positions
        }
        return batch

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

class FTQADataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, max_token_len, batch_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_data = pd.read_csv("{}/data/GT0/train_all_labeled.csv").format(cur_dir)
        self.val_data = pd.read_csv("{}/data/GT0/val_all_labeled.csv").format(cur_dir)
        self.test_data = pd.read_csv("{}/data/GT0/test_all_labeled.csv").format(cur_dir)
        self.max_token_len = max_token_len
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = FTQADataset(
            self.train_data,
            self.tokenizer,
            self.max_token_len
        )

        self.val_dataset = FTQADataset(
            self.val_data,
            self.tokenizer,
            self.max_token_len
        )
        
        self.test_dataset = FTQADataset(
            self.test_data,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            shuffle=True,
            num_workers=2,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=2,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=2,
            persistent_workers=True
        )
