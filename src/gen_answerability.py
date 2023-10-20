import ast
import numpy as np
from pathlib import Path
import math
import argparse
import os
import torch
import pytorch_lightning as pl
import pandas as pd
import sys
from tqdm import tqdm
from os.path import join
from torch.nn import functional as F
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from os import listdir
from os.path import isfile, join

cur_dir = os.path.dirname(os.path.realpath(__file__))
# setting path
sys.path.append('{}/Ans_eval_model/'.format(cur_dir))
from train1 import Model, MAX_TOKEN_COUNT, MODEL_NAME
from utils_ans import FTQADataModule, FTQADataset, trim_batch


TYPE2CONTROL_SIGNAL = ['What', 'How', 'Who', 'Why', 'Where', 'When', 'Which']

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--no_ans", default=-12, type=int)
parser.add_argument("--imp", default=0, type=int)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--epoch", type=int, default=32)
args, rest = parser.parse_known_args()

class Model_Pred(Model):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)                                                                                      
        self.no_ans_threshold = args.no_ans
        self.imp_threshold = args.imp
        self.tokenizer = tokenizer
        
    def test_step(self, batch, batch_idx: int):
        src = batch['input_ids']
        mask = batch['attention_mask']
        
        outputs = self(src, mask)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        answer_start = torch.topk(start_logits, 3)
        answer_end = torch.topk(end_logits, 3)        
        null_start = start_logits[:,0]
        null_end = end_logits[:,0]
        implicit_start = start_logits[:,1]
        implicit_end = end_logits[:,1]   
        
        preds = []
        for i in range(null_start.size()[0]):
            if answer_start.indices[i,0] == 0 and answer_start.indices[i,1] == 1:
                best_null_start = answer_start.indices[i,2]
            elif answer_start.indices[i,0] == 0 and answer_start.indices[i,1] != 1:
                best_null_start = answer_start.indices[i,1]
            elif answer_start.indices[i,0] == 1 and answer_start.indices[i,1] != 0:
                best_null_start = answer_start.indices[i,1]
            elif answer_start.indices[i,0] == 1 and answer_start.indices[i,1] == 0:
                best_null_start = answer_start.indices[i,2]                              
            else:
                best_null_start = answer_start.indices[i,0]
            if answer_end.indices[i,0] == 0 and answer_end.indices[i,1] == 1:
                best_null_end = answer_end.indices[i,2]
            elif answer_end.indices[i,0] == 0 and answer_end.indices[i,1] != 1:
                best_null_end = answer_end.indices[i,1]
            elif answer_end.indices[i,0] == 1 and answer_end.indices[i,1] != 0:
                best_null_end = answer_end.indices[i,1]
            elif answer_end.indices[i,0] == 1 and answer_end.indices[i,1] == 0:
                best_null_end = answer_end.indices[i,2]                              
            else:
                best_null_end = answer_end.indices[i,0]
            score_diff1 = null_start[i] + null_end[i] - \
                        start_logits[i,best_null_start] - end_logits[i,best_null_end] - self.no_ans_threshold
            score_diff2 = implicit_start[i] + implicit_end[i] - \
                        start_logits[i,best_null_start] - end_logits[i,best_null_end]                   
            if score_diff1 > 0 and score_diff1 > score_diff2:
                preds.append('')
            elif score_diff2 > self.imp_threshold:
                preds.append('<IMPLICIT>')         
            else:
                tokens = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(src[i, best_null_start:best_null_end+1]))
                preds.append(tokens)        
        return {"preds": preds}
        
    def test_epoch_end(self, outputs):
        preds = []
        for x in outputs: 
            preds += x['preds']
            
        context_num = len(DF['context'].unique())
        DF['answer_pred'] = preds
        df_dups_dropped = DF.drop_duplicates()
        a_pred = df_dups_dropped[~(df_dups_dropped.answer_pred == '')]
        implicit_pred = df_dups_dropped[df_dups_dropped.answer_pred == '<IMPLICIT>']
        a_pred = a_pred[~(a_pred.answer_pred == '<IMPLICIT>')]
        DF['answer_pred'] = preds
        implicit_pred = df_dups_dropped[df_dups_dropped.answer_pred == '<IMPLICIT>']
        a_pred = a_pred[~(a_pred.answer_pred == '<IMPLICIT>')]
        print('total data number:', len(DF))
        print('data number after drop duplicates:', len(df_dups_dropped))
        print('total data number per context:', len(DF)/context_num)
        print('answered number:', len(a_pred))
        print('implicit answer number:', len(implicit_pred))
        print('answered ratio:', (len(a_pred)+len(implicit_pred))/len(DF)*100)
        print('answered number per context:', (len(a_pred)+len(implicit_pred))/context_num)
               
        return preds


# Path(data_dir).mkdir(parents=True, exist_ok=True)
class GenDM(pl.LightningDataModule):
    def __init__(self, tokenizer, max_token_len, batch_size, data):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.batch_size = batch_size
        self.test_data = data
    def setup(self, stage=None):
        self.test_dataset = GenDS(
            self.test_data,
            self.tokenizer,
            self.max_token_len,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=20,
            persistent_workers=True
        )

class GenDS(FTQADataset, Dataset):
    def __init__(self, data, tokenizer, max_token_len):
        super(Dataset).__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_token_len
        self.df = data
        # self.df = self.df[self.df['qt'].isin(['<ACTION>', '<SETTING>', '<CHARACTER>', '<OUTCOME_RESOLUTION>'])].reset_index(drop=True)
        self.pad_token_id = self.tokenizer.pad_token_id

    def __getitem__(self, index):
        """Call tokenizer on src and tgt_lines"""
        source_line = self.df.loc[index, 'context']
        q_line = self.df.loc[index, 'tot_gen']
        source_inputs = self.tokenizer.encode_plus(['<IMPLICIT>'+ str(q_line), str(source_line)], max_length=self.max_source_length, 
                                    padding="max_length", return_tensors='pt', truncation=True)        
        source_ids = source_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
        }

    def collate_fn(self, batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        pad_token_id = self.pad_token_id

        src_ids, src_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": src_ids,
            "attention_mask": src_mask,
        }
        return batch

def run_generate(data, ckpt_path, gpus=0, bs=32):
    global DF
    DF = data.loc[:, ['context', 'tot_gen']]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<IMPLICIT>']})
    args.ckpt_path = ckpt_path
    model = Model_Pred(tokenizer)
    model = model.load_from_checkpoint(args.ckpt_path)
    print(args.ckpt_path)
    args.bs = bs
    data_module = GenDM(
        tokenizer,
        max_token_len=MAX_TOKEN_COUNT,
        batch_size=args.bs,
        data=DF,
        )
    trainer = pl.Trainer(accelerator='gpu', devices=[gpus])

    trainer.test(model, data_module)
    del model
    with torch.cuda.device('cuda:{}'.format(gpus)):
        torch.cuda.empty_cache()

def concat_gen(x, q_num):
    
    columns = ['Why']
    tot_gen = []
    for i in columns:
        tot_gen += x[i][:q_num]
        
    return tot_gen