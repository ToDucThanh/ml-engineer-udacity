import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, load_metric
from pytorch_lightning.callbacks import ModelCheckpoint

from smdebug import modes
import smdebug.pytorch as smd

class Pipeline_e2e(pl.LightningModule):
    def __init__(self, lr=2e-5, eps=1e-8, max_length=128, batch_size=32, num_workers=0, use_hook=False):
        super().__init__()
        # self.save_hyperparameters()
        self.lr = lr
        self.eps = eps
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_classes = 2
        self.num_workers = num_workers
        self.use_hook = use_hook
    
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=self.num_classes
        )
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.metric = load_metric("glue", "cola")
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.valid_loss_step = 0.0
        self.running_samples = 0
        
        self.valid_losses = []
        self.predictions = []
        self.references = []
        
        if use_hook:
            self.hook = smd.Hook.create_from_json_file()
            self.hook.register_hook(self.model)
            self.hook.register_loss(self.loss_fn)

    def prepare_data(self):
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]
        
        self.train_data = self.train_data.map(self.tokenize_sentence, batched=True)
        self.val_data = self.val_data.map(self.tokenize_sentence, batched=True)
        
        self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )
        self.val_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

    def tokenize_sentence(self, batch):
        outputs = self.tokenizer(
                    batch["sentence"],
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    )
        
        return outputs

    def forward(self, batch):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        return logits

    def train_dataloader(self):
        dataloader = DataLoader(self.train_data, 
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                num_workers=self.num_workers)           
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.train_data, 
                                    batch_size=self.batch_size, 
                                    shuffle=False, 
                                    num_workers=self.num_workers)
           
        return dataloader

    def training_step(self, batch, batch_idx):
        if self.use_hook:
            self.hook.set_mode(modes.TRAIN)
            
        labels = batch["label"]
        logits = self.forward(batch)
        loss = self.loss_fn(logits.view(-1, self.num_classes), labels)
        
        self.log("train_loss", loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        if self.use_hook:
            self.hook.set_mode(modes.EVAL)
            
        labels = batch["label"]
        logits = self.forward(batch)
        loss = self.loss_fn(logits.view(-1, self.num_classes), labels)
        preds = torch.argmax(logits, dim=1)
        
        self.log("valid_loss", loss, prog_bar=True)
        
        self.valid_loss_step += loss.item()
        self.running_samples += len(labels)
        
        self.predictions.append(preds)
        self.references.append(labels)

    def on_validation_epoch_end(self):
        predictions = torch.concat(self.predictions).view(-1)
        references = torch.concat(self.references).view(-1)
        matthews_correlation = self.metric.compute(
            predictions=predictions, references=references
        )
        
        valid_loss = self.valid_loss_step/self.running_samples
        self.valid_losses.append(valid_loss)
        
        self.log_dict(matthews_correlation, sync_dist=True, prog_bar=True)
        
        self.predictions.clear()
        self.references.clear()
        self.valid_loss_step = 0.0
        self.running_samples = 0
        
    def on_train_epoch_end(self):
        valid_loss_1_training_epoch = np.mean(self.valid_losses)
        self.valid_losses.clear()
        print("\nValidation set: Average loss: {:.4f}\n".format(valid_loss_1_training_epoch))
              
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, eps=self.eps)