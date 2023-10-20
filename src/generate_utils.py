from typing import Dict, List, Tuple
import torch
import joblib
import os
from os.path import join
import pandas as pd
import numpy as np
from functools import partial
from itertools import repeat

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_summaries_examples(
    examples: List[Dict],
    batch_num,
    model,
    qt,
    device: str = DEFAULT_DEVICE,
    args=None,
    **generate_kwargs,
) -> Dict:
    global QT
    QT = qt
    qt_tokens = model.tokenizer(qt, return_tensors="pt", truncation=True, padding="longest").to('cuda:{}'.format(args.device))
    add_qt_context = lambda x, y :  y + '<SEP>' + x + '<SEP>'
    examples = list(add_qt_context(examples, qt))
    batch = model.tokenizer(examples, return_tensors="pt", 
                        truncation=True, padding="longest").to('cuda:{}'.format(args.device))

    if batch.input_ids.shape[1] > 900:
        batch.input_ids = batch.input_ids[:, :900]
        batch.attention_mask = batch.attention_mask[:, :900]
    temp_tgt = []
    generated_sents = []
    summaries = model.generate(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        qt=qt_tokens.input_ids,        
        temp_tgt=temp_tgt,
        **generate_kwargs,
    )
    generated_sent = model.tokenizer.batch_decode(summaries, skip_special_tokens=True,clean_up_tokenization_spaces=False)
    tgt_input_ids, tgt_masks = model.tokenizer(generated_sent, return_tensors="pt", truncation=True, padding="longest").to('cuda:{}'.format(device)).values()
    temp_tgt = [tgt_input_ids, tgt_masks]
    for i in generated_sent:
        generated_sents.append([i])
    for i in range(1, args.generate_num):
        summaries = model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            qt=qt_tokens.input_ids,  
            temp_tgt=temp_tgt,
            **generate_kwargs,
        )
        temp = model.tokenizer.batch_decode(summaries, skip_special_tokens=True,clean_up_tokenization_spaces=False)
        
        cand_group = []
        for j in range(len(examples)):
            cand = check_dups(generated_sents[j], temp[j*(int(len(temp)/len(examples))): (j+1)*(int(len(temp)/len(examples)))])
            generated_sents[j].append(cand)
            cand_group.append(cand)
        tgt_temp_ids, tgt_temp_masks = model.tokenizer(cand_group, return_tensors="pt", truncation=True, padding="longest").to('cuda:{}'.format(device)).values()
        tgt_input_ids = torch.cat((tgt_input_ids, tgt_temp_ids), dim=1)
        tgt_masks = torch.cat((tgt_masks, tgt_temp_masks), dim=1)
        temp_tgt = [tgt_input_ids, tgt_masks]
    # print(generated_sents[j])
    return generated_sents

def check_dups(gen_list, gen):
    for i in gen:
        dups = False
        for j in gen_list:
            if i == j:
                dups = True
                continue
        if dups == False:
            return i
    return gen_list[0]

def load_data(data_dir):
    try:
        df = pd.read_excel(data_dir)
        df = df.loc[:,['cor_section', 'question']]
        df.columns = ['context', 'question']
    except:
        df = pd.read_csv(data_dir)
        df = df.loc[:,['cor_section', 'question']]
        df.columns = ['context', 'question']
    
    return df