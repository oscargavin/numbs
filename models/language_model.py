import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from config import Config

class LanguageModel(nn.Module):
    def __init__(self, model_path="./gpt2_finetuned_captioner"):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=Config.MAX_LENGTH)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)