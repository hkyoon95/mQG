import os
from torch import optim, nn, utils, Tensor
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from utils_ans import FTQADataset, FTQADataModule


cur_dir = os.path.dirname(os.path.realpath(__file__))

RANDOM_SEED = 42
MAX_TOKEN_COUNT = 768
N_EPOCHS = 7
BATCH_SIZE = 32
GPUS = [0]
MODEL_NAME = "deepset/deberta-v3-base-squad2"
LR = 5e-6
pl.seed_everything(RANDOM_SEED)

# define the LightningModule
class Model(pl.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
        self.model.config.max_position_embeddings = MAX_TOKEN_COUNT
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.step_count = 0
        self.save_hyperparameters()
        self.warmup_steps = 0
        self.total_training_steps = 0

    def forward(self, src, mask, start_positions=None, end_positions=None):
        outputs = self.model(src, mask, start_positions=start_positions, end_positions=end_positions)

        return outputs

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        src = batch['input_ids']
        mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        
        outputs = self(src, mask, start_positions, end_positions)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        # CrossEntropyLoss
        loss = outputs[0]
        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):

        src = batch['input_ids']
        mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        outputs = self(src, mask, start_positions, end_positions)
        loss = outputs[0]
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=LR)
        scheduler = ReduceLROnPlateau(optimizer, patience=1, verbose=True, factor=0.2)
        lr_scheduler = {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}
        return [optimizer], [lr_scheduler]
        # return [optimizer]

from argparse import ArgumentParser
  
def main(hparams):

    # init the automodel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<IMPLICIT>']})
    model = Model(tokenizer)
    logger = TensorBoardLogger("{}/lr{}".format(cur_dir, LR))
    data_module = FTQADataModule(
    tokenizer,
    batch_size=BATCH_SIZE,
    max_token_len=MAX_TOKEN_COUNT
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='{}/lr{}'.format(cur_dir, LR), 
        monitor="val_loss", 
        mode="min", 
        save_top_k=20,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(max_epochs=N_EPOCHS,
                    logger=logger, 
                    accelerator="gpu", gpus=GPUS, strategy="dp",
                    callbacks=[checkpoint_callback, lr_monitor],
                    val_check_interval=0.5,
                    # auto_lr_find=True
    )
    # lr_finder = trainer.tuner.lr_find(model)
    trainer.fit(model, data_module)
  
if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
  
    main(args)