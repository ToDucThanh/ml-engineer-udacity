import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification 



class Net(nn.Module):
    def __init__(self, device):
        
        super().__init__()
    
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")     
        
    def tokenize_sentence(self, batch):
        outputs = self.tokenizer(
                    batch["sentence"],
                    max_length=256,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                    )
        
        return outputs
    
    def forward(self, sample):
        processed_sample = self.tokenize_sentence(sample)
        input_ids, attention_mask = processed_sample["input_ids"], processed_sample["attention_mask"]
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        return logits