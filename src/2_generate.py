import json
import time
import warnings
import re
from logging import getLogger
import joblib
import time

import argparse
import glob
import logging
import os
from os.path import join
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import torch
from tqdm import tqdm
from lightning_base import BaseTransformer, add_generic_args, generic_train
from transformers.models.bart.modeling_bart import shift_tokens_right

from generate_utils import generate_summaries_examples, load_data 
from utils import freeze_params, lmap, use_task_specific_params, parse_numeric_cl_kwargs

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, help="like cnn_dm/test.source")
parser.add_argument("--save_path", type=str, help="where to save summaries")
parser.add_argument("--score_path", type=str, required=False, default="metrics.json", help="where to save metrics")
parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
parser.add_argument("--prefix", type=int, required=False, default=2, help="will be added to the begininng of src examples")
parser.add_argument("--task", type=str, default="summarization", help="used for task_specific_params + metrics")
parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
parser.add_argument("--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all.")
parser.add_argument("--fp16", action="store_true")
################################
parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
parser.add_argument(
        "--ckpt_path",
        default=None,
        type=str,
        help='path tooo stored model checkpoints',
    )
parser.add_argument("--output_dir", type=str)
parser.add_argument("--model_name_or_path", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
parser.add_argument("--config_name", type=str)
parser.add_argument("--tokenizer_name", type=str)
parser.add_argument("--test_max_target_length", type=int)
parser.add_argument("--eval_max_length", type=int)
parser.add_argument("--beam", default=5, type=int)
parser.add_argument("--search_mode", default='beam', type=str)
parser.add_argument("--data", default='test', type=str, help='data type')
parser.add_argument("--data_dir", type=str, help='data dir')
parser.add_argument("--generate_num", type=int)

################################
args, rest = parser.parse_known_args()
# print(rest)
parsed = parse_numeric_cl_kwargs(rest)

TYPE2CONTROL_SIGNAL = ['What', 'How', 'Who', 'Why', 'Where', 'When', 'Which']

class mQG(BaseTransformer):
    mode = "mQG"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        use_task_specific_params(self.model, self.mode)
        self.last_hidden_states = None
        self.generate_kwargs = {}
    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def generate(self, input_ids, attention_mask, qt,  temp_tgt, num_q=args.beam, early_stop=True, **generate_kwargs):


        # prefix for decoder
        qt = shift_tokens_right(qt, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)
        ones = torch.ones(len(input_ids), qt.size()[1]).cuda(qt.get_device())
        prefix = (ones * qt).type(torch.int64)
        prefix = prefix[:, :args.prefix+1]
        beam = args.beam           
        
        if temp_tgt == []:
            generate_kwargs.update(decoder_input_ids=prefix)
            
            if args.search_mode == 'beam':
                generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        num_beams=beam,
                        max_length=args.test_max_target_length,
                        min_length=1,
                        repetition_penalty=1.5,
                        length_penalty=3,
                        early_stopping=True,
                        use_cache=False,
                        num_return_sequences=1,
                        **generate_kwargs
                    )
            return generated_ids           
        else:
            src_q_ids = torch.cat([input_ids, temp_tgt[0]], dim=1)
            src_q_masks = torch.cat([attention_mask, temp_tgt[1]], dim=1)
            generate_kwargs.update(decoder_input_ids=prefix)
            if args.search_mode == 'beam':
                generated_ids = self.model.generate(
                        input_ids=src_q_ids,
                        attention_mask=src_q_masks,
                        num_beams=beam,
                        max_length=args.test_max_target_length,
                        min_length=1,
                        repetition_penalty=1.5,
                        length_penalty=3,
                        early_stopping=True,
                        use_cache=False,
                        num_return_sequences=beam,
                        **generate_kwargs
                    )
            return generated_ids              

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory
    except OSError:
        print("OSError")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def run_generate():    
    model: mQG = mQG(args)
    model = model.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.to('cuda:{}'.format(args.device))
    
    print('#############################################')
    print("# model is loaded from", args.ckpt_path)
    print('# tokenizer.all_special_tokens =', model.tokenizer.all_special_tokens)
    print('# tokenizer.all_special_ids =', model.tokenizer.all_special_ids)
    print('#############################################')
    # update config with task specific params
    use_task_specific_params(model, 'summarization')
        
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result_dir= os.path.join(args.output_dir, 'pred_q')        
    df = load_data(args.data_dir)

    df_context = df['context'].unique().copy()
    tot_batch = round(len(df_context)/args.bs)
    result_df = pd.DataFrame(columns = TYPE2CONTROL_SIGNAL)
    result_df['context'] = df_context
    gen_dict = {qt: [] for qt in TYPE2CONTROL_SIGNAL}
    for i in range(tot_batch):
        if i != tot_batch - 1:
            examples = df_context[i*args.bs: (i+1)*args.bs]
          
            for qt in TYPE2CONTROL_SIGNAL:
                generated_sents = generate_summaries_examples(
                    examples,
                    len(examples),
                    model,
                    qt,
                    device=args.device,
                    args=args,
                    **parsed,
                )
                gen_dict[qt] += generated_sents
        else:
            examples = df_context[(tot_batch - 1)*args.bs:]  
            for qt in TYPE2CONTROL_SIGNAL: 
                generated_sents = generate_summaries_examples(
                    examples,
                    len(examples),
                    model,
                    qt,
                    device=args.device,
                    args=args,
                    **parsed,
                )
                gen_dict[qt] += generated_sents
        print((i+1)/tot_batch*100)    
    gen_dict = pd.DataFrame(gen_dict)
    result_df = pd.concat([result_df['context'], gen_dict], axis=1)
    del model
    with torch.cuda.device('cuda:{}'.format(args.device)):
        torch.cuda.empty_cache()
    result_df.to_csv(result_dir+'.csv', index=False)
    
if __name__ == "__main__":
    run_generate()
