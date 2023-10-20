import argparse
from cmath import nan
import glob
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_base import BaseTransformer, add_generic_args, generic_train
from transformers import MBartTokenizer, T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right

logger = logging.getLogger(__name__)

from argparse import Namespace
from mqs_loss import mqs_loss
try:
    from .callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
    from .utils import (
        ROUGE_KEYS, LegacySeq2SeqDataset, assert_all_frozen, calculate_bleu, calculate_rouge, flatten_list,
        freeze_params, get_git_info, label_smoothed_nll_loss, lmap, pickle_save, save_git_info, save_json,
        use_task_specific_params,
    )
except ImportError:
    from callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
    from utils import (
        ROUGE_KEYS, LegacySeq2SeqDataset, assert_all_frozen,
        calculate_bleu, calculate_rouge, flatten_list, freeze_params, get_git_info, label_smoothed_nll_loss,
        lmap, pickle_save, save_git_info, save_json, use_task_specific_params,
    )

cur_dir = os.path.dirname(os.path.realpath(__file__))

class mQG(BaseTransformer):
    mode = "mQG"
    loss_names = ["loss", "mle_loss", "mqs_loss"]

    def __init__(self, hparams, **kwargs):
        ############################
        if 'sortish_sampler' not in hparams:
            hparams['sortish_sampler'] = False
            hparams = Namespace(**hparams)
        if 'num_workers' not in hparams:
            hparams['num_workers'] = 16
            hparams = Namespace(**hparams)
        if type(hparams) != argparse.Namespace:
            hparams = Namespace(**hparams)
        ############################
        if hparams.sortish_sampler and hparams.gpus > 1:
            hparams.replace_sampler_ddp = False
        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
   
        use_task_specific_params(self.model, self.mode)
#         save_git_info(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
#         pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)
        # ========= TIMER ==========
        self.timer_count = 0
        self.timer_sum = 0
        self.random_seed = 0
        self.dataset_kwargs: dict = dict(
            random_seed = self.random_seed,
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
            path_or_data=self.hparams.path_or_data,
            ori_dir=self.hparams.ori_dir
            #######################
            # fuse_num=self.hparams.fuse_num,
            # type_embedding=self.hparams.type_embedding,
            #######################
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"

        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())
        
        # additional layer

#       self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.decoder_start_token_id = None  # default to config
        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams.tgt_lang]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id
        self.dataset_class = (LegacySeq2SeqDataset)

        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        assert self.eval_beams >= 1, f"got self.eval_beams={self.eval_beams}. Need an integer > 1"
        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length
        self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric

        self.ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

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
    
    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)    
        
    def forward(self, input_ids, seqs_num, src_mask, decoder_input_ids, decoder_masks, use_cache, seqs_masks, context_len, q_len, bs):
        
        src_last_hidden_state = self.model.get_encoder()(input_ids, attention_mask=src_mask).last_hidden_state
        tgt_sent_last_hidden_states = []
        hid_dim = src_last_hidden_state.shape[-1]
        # get sentence representation before targets
        if max(seqs_num) != 0:
            q_last_hidden_state = src_last_hidden_state[:,context_len:,:]
            q_last_hidden_state = q_last_hidden_state.reshape(bs, -1, q_len, hid_dim)
            for i in range(max(seqs_num)):
                temp = q_last_hidden_state[:,i,:,:]
                # pooling
                # expand to hidden_size
                input_mask_expanded = seqs_masks[:,i,:].unsqueeze(-1).expand(temp.size()).float()
                sum_embeddings = torch.sum(temp * input_mask_expanded, 1)
                # normalize
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                sum_embeddings = (sum_embeddings / sum_mask).unsqueeze(1)
                tgt_sent_last_hidden_states.append(sum_embeddings)
            tgt_sent_last_hidden_states = torch.cat(tgt_sent_last_hidden_states, dim=1)

        outputs = self.model.get_decoder()(decoder_input_ids, encoder_hidden_states=src_last_hidden_state, use_cache=use_cache)
        logits = self.model.lm_head(outputs[0])

        if max(seqs_num) != 0:
            last_hidden_states = outputs.last_hidden_state
            decoder_mask_expanded = decoder_masks.unsqueeze(-1).expand(last_hidden_states.size()).float()
            # qt_mask = torch.zeros(decoder_mask_expanded.size()).cuda(decoder_mask_expanded.get_device())
            # qt_mask[:, 1, :] = -1
            # decoder_mask_expanded = decoder_mask_expanded + qt_mask
            sent_hidden_states = torch.sum(last_hidden_states * decoder_mask_expanded, 1)
            # normalize
            decoder_mask = input_mask_expanded.sum(1)
            decoder_mask = torch.clamp(decoder_mask, min=1e-9)
            sent_hidden_states = (sent_hidden_states / decoder_mask).unsqueeze(1)
            sentloss_hidden_states = torch.cat((sent_hidden_states, tgt_sent_last_hidden_states), dim=1)
            norm = sentloss_hidden_states.norm(dim=2, keepdim=True)
            # due to seq padding
            norm[norm == 0.] = 1
            norm_rep = sentloss_hidden_states / norm
            cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1,2))
            sentloss = mqs_loss(cosine_scores, 
                                 seqs_num, prefix_len=0)            
            output = (logits, outputs.last_hidden_state, sentloss)
        else:
            sentloss = None
            output = (logits, outputs.last_hidden_state, sentloss)
        
        return output

    def _step(self, batch: dict) -> Tuple:
        cls_token_id = self.tokenizer.cls_token_id
        pad_token_id = self.tokenizer.pad_token_id
        decoder_start_token_id = self.tokenizer.eos_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids, tgt_mask = batch["labels"], batch["labels_mask"]
        seqs_num = batch["seqs_num"]
        seqs_ids = batch['seqs_ids']
        seqs_masks = batch['seqs_masks']
        decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id, decoder_start_token_id)
        # ========== TIMER ===========
        # change remain eos token to pad token
        index_of_eos = (decoder_input_ids.ne(pad_token_id).sum(dim=1) - 1) 
        decoder_input_ids[range(decoder_input_ids.shape[0]), index_of_eos] = 1
        bs = src_ids.shape[0]
        context_len = src_ids.shape[1]
        if max(seqs_num) != 0:
            q_len = seqs_ids.shape[2]
            seqs_ids = seqs_ids.reshape(bs, -1)
            seqs_total_masks = seqs_masks.reshape(bs, -1)
            src_q_ids = torch.cat([src_ids,seqs_ids], dim=1)
            src_q_mask = torch.cat([src_mask,seqs_total_masks], dim=1)
            outputs = self(src_q_ids, seqs_num, src_q_mask, decoder_input_ids, tgt_mask, False, seqs_masks, context_len, q_len, bs)
        else:
            q_len = torch.tensor(0).to(src_ids.device)
            outputs = self(src_ids, seqs_num, src_mask, decoder_input_ids, tgt_mask, False, seqs_masks, context_len, q_len, bs)              
        lm_logits = outputs[0]
        mqs_loss = outputs[2]
        mle_loss = self.ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        if self.hparams.loss_n == 'ce_mqs':
            if mqs_loss != None:
                loss = mle_loss + self.hparams.beta * mqs_loss
            else:
                loss = mle_loss
                mqs_loss = torch.zeros(mle_loss.size()).cuda(mle_loss.get_device())
        if self.hparams.loss_n == 'ce_mqs_nobeta':
            if mqs_loss != None:
                loss = mle_loss + mqs_loss
            else:
                loss = mle_loss
                mqs_loss = torch.zeros(mle_loss.size()).cuda(mle_loss.get_device())                            
        elif self.hparams.loss_n == 'onlyce':
            loss = mle_loss
            if mqs_loss == None:
                mqs_loss = torch.zeros(mle_loss.size()).cuda(mle_loss.get_device())              

        loss = loss.mean()     
        return (loss, mle_loss, mqs_loss) 


    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        
        return base_metrics

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        for k in logs.keys():
            self.log(k, logs[k], prog_bar=True, logger=True)

        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        generative_metrics = {k: v.item() for k, v in losses.items()}
        losses.update(generative_metrics)
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count
        self.save_metrics(all_metrics, prefix)
        for k in all_metrics.keys():
            self.log(k, all_metrics[k], on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", all_metrics["val_avg_mle_loss"] + all_metrics["val_avg_mqs_loss"], on_epoch=True, prog_bar=True, logger=True)
        return {
            f"{prefix}_loss": loss,
        }

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path):
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    ##############################
    # def get_dataset(self, type_path) -> Seq2SeqDatasetForFID:
    #     n_obs = self.n_obs[type_path]
    #     max_target_length = self.target_lens[type_path]
    #     dataset = self.dataset_class(
    #         self.tokenizer,
    #         type_path=type_path,
    #         n_obs=n_obs,
    #         max_target_length=max_target_length,
    #         **self.dataset_kwargs,
    #     )
    #     return dataset
    ##############################

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)
        sampler = None
        if self.hparams.sortish_sampler and type_path == "train":
            sampler = dataset.make_sortish_sampler(batch_size, distributed=self.hparams.gpus > 1)
            shuffle = False

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            sampler=sampler,
            worker_init_fn=np.random.seed(self.hparams.seed)
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        self.random_seed += 1
        self.dataset_kwargs['random_seed'] = self.random_seed
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=512,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=142,  # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=142,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument(
            "--task", type=str, default="summarization", required=False, help="# examples. -1 means use all."
        )
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)
        parser.add_argument("--eval_beams", type=int, default=None, required=False)
        parser.add_argument(
            "--val_metric", type=str, default='val_avg_loss', required=False, choices=["bleu", "rouge2", "loss", "rougel",None]
        )
        parser.add_argument("--eval_max_gen_length", type=int, default=None, help="never generate more than n tokens")
        parser.add_argument("--save_top_k", type=int, default=-1, required=False, help="How many checkpoints to save")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        ############################
        parser.add_argument(
            "--ckpt_path",
            default=None,
            type=str,
            help='path tooo stored model checkpoints',
        )
        parser.add_argument(
            "--fuse_num",
            default=None,
            type=int,
            help='num of passage vector to fuse in decoder',
        )
        parser.add_argument(
            "--type_embedding",
            action="store_true",
            help='whether to add a type embedding layer during encoding',
        )
        parser.add_argument("--attribute")
        parser.add_argument("--path_or_data", type=str, default='data', help='path or data')
        ############################
        parser.add_argument("--reload_data", type=bool, default=True)
        return parser
    
def main(args, model=None) -> mQG:
    if os.path.isdir(args.output_dir) and args.do_train:
        import shutil
        shutil.rmtree(args.output_dir)
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if model is None:
        if "summarization" in args.task:
            model: mQG = mQG(args)
    #################
    if args.ckpt_path is not None:
        model = model.load_from_checkpoint(
                    args.ckpt_path, 
                    data_dir=args.data_dir, 
                    output_dir=args.output_dir,
                    freeze_encoder=args.freeze_encoder,
                    max_epochs=args.max_epochs,
                    train_batch_size=args.train_batch_size,
                    eval_beams=args.eval_beams,
                    eval_batch_size=args.eval_batch_size,
                    fuse_num=args.fuse_num,
                    type_embedding=args.type_embedding,
                )
        if not args.freeze_encoder:
            for par in model.model.get_encoder().parameters():
                par.requires_grad = True
        print('******************************')
        print('Continue training from:', args.ckpt_path)
        # print('Parameters:', model.hparams)
        print('******************************')
    #################
    dataset = Path(args.data_dir).name
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        logger = TensorBoardLogger(cur_dir + "/lightning_logs", name=args.attribute)  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        project = os.environ.get("WANDB_PROJECT", dataset)
        logger = WandbLogger(name=model.output_dir.name, project=project)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset}")

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    else:
        es_callback = False

    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback= pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir, 
            monitor=model.val_metric, 
            mode="min", 
            save_top_k=args.save_top_k,
            filename='{epoch}'
        ),
        early_stopping_callback=es_callback,
        logger=logger,
        # accumulate_grad_batches=2,
        sync_batchnorm=True,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        # fast_dev_run=7,
        reload_dataloaders_every_epoch=args.reload_data,
    )
#     pickle_save(model.hparams, model.output_dir / "hparams.pkl")
    if not args.do_predict:
        return model

    model.hparams.test_checkpoint = ""
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
    trainer.logger.log_hyperparams(model.hparams)

    # test() without a model tests using the best checkpoint automatically
    # trainer.test()
    return model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = mQG.add_model_specific_args(parser, os.getcwd())
    import json

    args = parser.parse_args()
    # create text file
    
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    with open(args.data_dir + '/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(args.output_dir)   
    main(args)
    