import torch
import os

class Config:
    # Data
    DATA_DIR = "./data"
    BATCH_SIZE = 8
    NUM_WORKERS = 4

    # Model
    VISION_MODEL = "microsoft/swin-tiny-patch4-window7-224"
    VISION_MODEL_OUTPUT_SIZE = 768  # This should match the output size of your vision encoder
    LANGUAGE_MODEL = "gpt2"
    CAPTION_MODEL = "gpt2"
    NUM_UNFROZEN_LAYERS = 2
    IMAGE_SIZE = (224, 224)
    MAX_LENGTH = 512
    MAX_SEQ_LENGTH = 512
    PAD_TOKEN_ID = 50256
    PROJECTION_HEADS = 8

    # Training
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 10
    PATIENCE = 5
    ACCUMULATION_STEPS = 16
    EVAL_FREQUENCY = 1
    CLIP_GRAD_NORM = 1.0

    # GPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging
    LOG_INTERVAL = 15
    SAMPLE_INTERVAL = 100

    # Paths
    CACHE_DIR = './model_cache'
    CHECKPOINT_DIR = './checkpoints'

    # Mixed Precision
    USE_AMP = False

    # Model Fine-tuning
    NUM_UNFROZEN_LAYERS = 2

    # LoRA Configuration
    LORA_R = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["c_attn", "c_proj"]