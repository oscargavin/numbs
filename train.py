import os
from comet_ml import Experiment
import nltk
import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
import torch.cuda.amp as amp
from torch.amp import autocast
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from config import Config
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
from data_loader import get_data_loaders
from models.vision_encoder import VisionEncoder
from models.language_model import LanguageModel
from models.projection_layer import ProjectionLayer
from models.caption_generator import CaptionGenerator

def caption_loss(generated_logits, true_captions, tokenizer):
    loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
     
    # Tokenize the true captions
    true_caption_tokens = tokenizer(true_captions, padding='max_length', truncation=True, max_length=Config.MAX_SEQ_LENGTH, return_tensors="pt").input_ids.to(generated_logits.device)

    # Ensure that generated_logits and true_caption_tokens have the same sequence length
    min_length = min(generated_logits.size(1), true_caption_tokens.size(1))
    generated_logits = generated_logits[:, :min_length, :]
    true_caption_tokens = true_caption_tokens[:, :min_length]

    # Shift the generated_logits and true_caption_tokens
    shifted_logits = generated_logits[:, :-1, :].contiguous()
    shifted_labels = true_caption_tokens[:, 1:].contiguous()

    # Compute loss
    loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
    
    # Reshape loss to match the input shape
    loss = loss.view(shifted_labels.size())
    
    # Create a mask that gives higher weight to the EOS token
    eos_mask = (shifted_labels == tokenizer.eos_token_id).float()
    eos_weight = 2.0  # Adjust this value to change the emphasis on EOS token
    weighted_loss = loss * (1 + (eos_weight - 1) * eos_mask)
    
    # Add length penalty
    length_penalty = torch.exp(-0.1 * (shifted_labels != tokenizer.pad_token_id).float().sum(1))
    weighted_loss = weighted_loss * (1 + length_penalty.unsqueeze(1))

    return weighted_loss.mean()

def train_epoch(model, train_loader, optimizer, scaler, accumulation_steps, epoch, experiment):
    model.train()
    total_loss = 0
    valid_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for step, (images, captions) in enumerate(progress_bar):
        images = images.to(Config.DEVICE)
        
        with autocast('cuda', enabled=Config.USE_AMP):
            image_features = model['vision_encoder'](images)
            projected_features = model['projection_layer'](image_features)
            generated_logits = model['caption_generator'](projected_features, captions)

            loss = caption_loss(generated_logits, captions, model['caption_generator'].tokenizer)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.CLIP_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps
        valid_batches += 1

        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})

        # Log to Comet ML less frequently
        if step % Config.LOG_INTERVAL == 0:
            experiment.log_metric(f"train_loss_epoch_{epoch}", loss.item() * accumulation_steps, step=step)
            log_captions(generated_logits, captions, model['caption_generator'].tokenizer)

    return total_loss / valid_batches

def log_captions(generated_logits, true_captions, tokenizer):
    generated_tokens = torch.argmax(generated_logits, dim=-1)
    for i in range(min(len(true_captions), 5)):
        generated_caption = tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
        print(f"\nSample {i+1}:")
        print(f"Generated ({len(generated_caption.split())} words): {generated_caption}")
        print(f"True ({len(true_captions[i].split())} words): {true_captions[i]}")
        print("-" * 50)

def validate(model, val_loader, epoch, experiment):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for step, (images, captions) in enumerate(tqdm(val_loader, desc="Validating")):
            images = images.to(Config.DEVICE)
            
            with autocast('cuda', enabled=Config.USE_AMP):
                image_features = model['vision_encoder'](images)
                projected_features = model['projection_layer'](image_features)
                generated_logits = model['caption_generator'](projected_features, captions)

                loss = caption_loss(generated_logits, captions, model['caption_generator'].tokenizer)

            total_loss += loss.item()

            # Log to Comet ML
            experiment.log_metric(f"val_loss_epoch_{epoch}", loss.item(), step=step)

            if step % Config.LOG_INTERVAL == 0:
                log_captions(generated_logits, captions, model['caption_generator'].tokenizer)

    return total_loss / len(val_loader)

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, filename):
    checkpoint = {
        'epoch': epoch,
        'best_metric': best_metric,
        'model_state_dict': {
            'vision_encoder': model['vision_encoder'].state_dict(),
            'projection_layer': model['projection_layer'].state_dict(),
            'caption_generator': model['caption_generator'].model.state_dict(),  # Save LoRA weights
        },
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, filename)
    try:
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved successfully to {checkpoint_path}")
    except Exception as e:
        print(f"Failed to save checkpoint. Error: {str(e)}")

def get_optimizer(model):
    # Freeze the base model parameters
    for param in model['caption_generator'].model.parameters():
        param.requires_grad = False
    
    # Only train the LoRA parameters and the projection layer
    trainable_params = [
        {'params': model['caption_generator'].model.parameters(), 'lr': Config.LEARNING_RATE * 0.1},
        {'params': model['projection_layer'].parameters(), 'lr': Config.LEARNING_RATE}
    ]

    return AdamW(trainable_params, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

def train():

    # Create a Comet ML experiment
    experiment = Experiment(
        api_key="yncS2q9q4j8UPdveJP0yEmka2",
        project_name="coin_captioner",
        workspace="oscargavin",
    )

    train_loader, val_loader = get_data_loaders()
    
    vision_encoder = VisionEncoder().to(Config.DEVICE)
    language_model = LanguageModel().to(Config.DEVICE)
    projection_layer = ProjectionLayer(
        vision_dim=vision_encoder.model.config.hidden_size,
        language_dim=language_model.model.config.hidden_size
    ).to(Config.DEVICE)
    caption_generator = CaptionGenerator("./gpt2_finetuned_captioner").to(Config.DEVICE)

    model = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'language_model': language_model,
        'projection_layer': projection_layer,
        'caption_generator': caption_generator
    })

    optimizer = get_optimizer(model)
    scheduler = OneCycleLR(optimizer, max_lr=Config.LEARNING_RATE, 
                           steps_per_epoch=len(train_loader) // Config.ACCUMULATION_STEPS, 
                           epochs=Config.NUM_EPOCHS)
    
    scaler = amp.GradScaler(enabled=Config.USE_AMP)

    best_val_loss = float('inf')
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, Config.ACCUMULATION_STEPS, epoch, experiment)
        
        if epoch % Config.EVAL_FREQUENCY == 0:
            val_loss = validate(model, val_loader, epoch, experiment)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Log epoch-level metrics
            experiment.log_metric("train_loss", train_loss, epoch=epoch)
            experiment.log_metric("val_loss", val_loss, epoch=epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, 'best_model.pth')

        scheduler.step()
        torch.cuda.empty_cache()

    # End the Comet ML experiment
    experiment.end()

if __name__ == "__main__":
    train()