from models.projection_layer import ProjectionLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from models.language_model import LanguageModel
from peft import get_peft_model, LoraConfig, TaskType
from config import Config

class CaptionGenerator(nn.Module):
    def __init__(self, custom_tokenizer_path="./gpt2_finetuned_captioner"):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(custom_tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = LanguageModel(custom_tokenizer_path)
        
        # Configure LoRA (if you still want to use it)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            target_modules=Config.LORA_TARGET_MODULES
        )
        
        # Apply LoRA to the model
        self.model.model = get_peft_model(self.model.model, peft_config)
        
        self.img_projection = ProjectionLayer(Config.VISION_MODEL_OUTPUT_SIZE, self.model.model.config.n_embd)

    def forward(self, projected_features, captions=None):
        batch_size = projected_features.size(0)
        device = projected_features.device
        
        # Project image features
        img_features = self.img_projection(projected_features)
        
        # Ensure img_features has 3 dimensions (batch_size, 1, hidden_size)
        if img_features.dim() == 2:
            img_features = img_features.unsqueeze(1)
        elif img_features.dim() == 3:
            img_features = img_features.mean(dim=1, keepdim=True)
        
        # Use a separator token if available, otherwise use the EOS token
        if self.tokenizer.additional_special_tokens:
            sep_token_id = self.tokenizer.additional_special_tokens_ids[0]
        else:
            sep_token_id = self.tokenizer.eos_token_id
        
        # Use the EOS token as a separator
        sep_token_id = self.tokenizer.eos_token_id
        sep_token_embed = self.model.model.transformer.wte(torch.tensor([sep_token_id], device=device)).unsqueeze(0).repeat(batch_size, 1, 1)

        if captions is not None:
            # Training mode
            max_caption_length = Config.MAX_SEQ_LENGTH - img_features.size(1) - 1
            caption_input = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=max_caption_length).input_ids.to(device)
            
            # Concatenate image features with separator token and caption embeddings
            text_embeds = self.model.model.transformer.wte(caption_input)
            inputs_embeds = torch.cat([img_features, sep_token_embed, text_embeds], dim=1)
        else:
            # Generation mode
            input_ids = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device)
            inputs_embeds = torch.cat([img_features, sep_token_embed, self.model.model.transformer.wte(input_ids)], dim=1)

        # Create attention mask
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        
        # Prepare position IDs
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
            
        outputs = self.model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        # Ensure logits have the expected shape
        logits = outputs.logits
        if captions is not None:
            # Truncate logits to match the expected sequence length
            expected_length = inputs_embeds.size(1)
            logits = logits[:, :expected_length, :]
        
        return logits
        
    def generate(self, projected_features, max_length=Config.MAX_SEQ_LENGTH):
        batch_size = projected_features.size(0)
        device = projected_features.device
        img_features = self.img_projection(projected_features)
        
        # Prepare input: <|img|> token followed by BOS token
        img_token = self.tokenizer.additional_special_tokens_ids[0]
        input_ids = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device)
        
        # Create inputs_embeds for the image features
        inputs_embeds = img_features.unsqueeze(1)
        
        # Ensure all tensors are on the same device
        self.model.model.to(device)
        
        # Generate
        output = self.model.model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            max_length=max_length,
            min_length=10,
            num_beams=Config.PROJECTION_HEADS,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.5,
            length_penalty=1.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        
        return output