import pandas as pd
import os
import re
import sys
from evaluate import load
from ignite.metrics import Bleu
from torchmetrics.text.rouge import ROUGEScore
from typing import Callable, Dict, Iterable, List, Union, Tuple
from os import listdir
from os.path import isfile, join
from pathlib import Path
from gen_answerability import run_generate


def concat_gen(x, q_num, TYPE2CONTROL_SIGNAL):
    tot_gen = []
    for i in TYPE2CONTROL_SIGNAL:
        tot_gen += x[i][:q_num]
    return tot_gen

def tokenize(x):
    temp = []
    for i in x:
        try:
            temp.append(i.split())
        except:
            continue
    return temp

def rougel_eval(df):
    rouge = ROUGEScore(use_stemmer=True, rouge_keys='rougeL', accumulate='best')
    
    for j in df.index:
        for c in df.loc[j, 'question']:
            rouge(c, [df.loc[j, 'tot_gen']])
            
    rougel_list = rouge.compute()
    print('rouge-L:', rougel_list)
    
    return rougel_list

def selfbleu_eval(df):
    m = Bleu(smooth='smooth1')

    for j in df.index:
        tokenized_preds = tokenize(df.loc[j, 'tot_gen'])
        for p in range(len(df.loc[j, 'tot_gen'])):
            temp = tokenized_preds.copy()
            temp.pop(p)
            m.update(([tokenized_preds[p]], [temp]))
            
    bleu_list = m.compute().item()
    print('Self-BLEU:', bleu_list)
    
    return bleu_list

def bertscore_eval(df):
    bertscore = load("bertscore")
    results = []

    for j in df.index:
        for c in df.loc[j, 'question']:
            references = [c for i in range(len(df.loc[j, 'tot_gen']))]
            predictions = df.loc[j, 'tot_gen']
            result = bertscore.compute(predictions=predictions, references=references, model_type="roberta-large", device='cuda:0')
            results.append(max(result['f1']))
            
    score = sum(results) / len(results)        
    print('BERScore:', score)
    return score
    
def bleurt_eval(df):
    bleurt = load("bleurt", "bleurt-20", module_type="metric")
    results = []

    for j in df.index:
        for c in df.loc[j, 'question']:
            references = [c for i in range(len(df.loc[j, 'tot_gen']))]
            predictions = df.loc[j, 'tot_gen']
            result = bleurt.compute(predictions=predictions, references=references)
            results.append(max(result["scores"]))

    score = sum(results) / len(results)        
    print('BLEURT:', score)
    return score

def evaluate(gold_df, result_df, ckpt_path, TYPE2CONTROL_SIGNAL, q_num):
    
    gb = gold_df.groupby(['context'])
    result = pd.DataFrame(gb['question'].unique())

    tot_df = pd.merge(result, result_df, on="context")
    tot_df = tot_df.dropna()
    tot_df['tot_gen'] = tot_df.apply(lambda x: concat_gen(x, q_num, TYPE2CONTROL_SIGNAL), axis=1)

    ans_df = tot_df.loc[:, ['context', 'tot_gen']]
    ans_df = ans_df.explode('tot_gen').reset_index(inplace=False, drop=True)
    
    run_generate(ans_df, ckpt_path)