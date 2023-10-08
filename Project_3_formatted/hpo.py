import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import smdebug.pytorch as smd

from datasets import load_from_disk, load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
from smdebug import modes


class GrammarClassification:
    def __init__(self, args):
        
        super().__init__()
        self.dataset_directory = args.training_input
        self.model_save_path = args.model_dir
        self.lr = args.lr
        self.eps = args.eps
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_classes = 2
    
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def setup(self, device, use_hook=False):
        self.hook = None
        self.model = self.model.to(device)
        self.metric = load_metric("glue", "cola")
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, eps=self.eps)
        
        if use_hook:
            self.hook = smd.Hook.create_from_json_file()
            self.hook.register_hook(self.model)
            self.hook.register_loss(self.loss_fn)        
            
    def prepare_data(self):
        cola_dataset = load_from_disk(self.dataset_directory)
        train_data = cola_dataset["train"]
        val_data = cola_dataset["validation"]
        
        train_data = train_data.map(self.tokenize_sentence, batched=True)
        val_data = val_data.map(self.tokenize_sentence, batched=True)
        
        train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
        train_loader = DataLoader(
            train_data, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers)
        
        val_loader = DataLoader(
            val_data, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers)
        
        return train_loader, val_loader
           
    def tokenize_sentence(self, batch):
        outputs = self.tokenizer(
                    batch["sentence"],
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    )
        
        return outputs
    
    def train(self, train_loader, epoch, device):
        self.model.train()
        if self.hook is not None:
            self.hook.set_mode(modes.TRAIN)
            
        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["label"]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = self.loss_fn(logits.view(-1, self.num_classes), labels)
            loss.backward()
            self.optimizer.step()
            
            _, preds = torch.max(logits, 1)
            running_loss += float(loss.item() * len(input_ids))
            running_corrects += float(torch.sum(preds == labels.data))
            running_samples += len(input_ids)
            accuracy = float(running_corrects)/float(running_samples) * 100
            
            print(
                "Train set: Epoch: {} [({:.0f}%)]\tLoss: {:.4f}\tAccuracy: {:.2f}%\n \
                Hyperparameters: Learning rate: {}, Batch size: {}, Epsilon: {}".format(
                    epoch,
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    accuracy,
                    self.lr, 
                    self.batch_size,
                    self.eps
                )
            )
            
    def test(self, val_loader, epoch, device):
        self.model.eval()
        if self.hook is not None:
            self.hook.set_mode(modes.EVAL)
        
        val_loss = 0
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader): 
                input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["label"]
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device) 
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = self.loss_fn(logits.view(-1, self.num_classes), labels)
                preds = torch.argmax(logits, dim=1)
                val_loss += loss.item()
                
                predictions.append(preds)
                references.append(labels)
                
            val_loss /= len(val_loader.dataset)
            predictions = torch.concat(predictions).view(-1)
            references = torch.concat(references).view(-1)
            matthews_correlation = self.metric.compute(predictions=predictions, references=references)
            print("\nValidation set: Average loss: {:.4f}\n".format(val_loss))
            print("Mathhews correlation coefficient: ", matthews_correlation)
            
    def save_model(self):
        with open(os.path.join(self.model_save_path, 'model.pth'), 'wb') as f:
            torch.save(self.model.state_dict(), f)


def main():
    parser = argparse.ArgumentParser(description="Grammar Classification")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=2e-5, 
        help="learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--eps", 
        type=float, 
        default=1e-8, 
        help="epsilon (default: 1e-8)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        metavar="N",
        help="The maximum length to truncate a sentence (default: 512)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        metavar="N",
        help="The number of workers used for dataloader (default: 0)"
    )
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument(
        "--training-input",
        type=str,
        default=os.environ["SM_CHANNEL_TRAINING"]
    )
    
    args = parser.parse_args()
    
    classifier = GrammarClassification(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup
    classifier.setup(device, use_hook=False)
    
    # Create train dataloder and validation dataloader
    train_loader, val_loader = classifier.prepare_data()
    
    # Train classifier
    for epoch in range(1, args.epochs + 1):
        classifier.train(train_loader, epoch, device)
        classifier.test(val_loader, epoch, device)
        
    # Save classifier
    classifier.save_model()
    
       
if __name__ == "__main__":
    main()