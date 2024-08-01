import torch
import torch.nn as nn
from transformers import AutoImageProcessor, SwinModel
from config import Config

def freeze_layers(model, num_layers_to_unfreeze):
    for param in model.parameters():
        param.requires_grad = False

    for param in list(model.parameters())[-num_layers_to_unfreeze * 2:]:
        param.requires_grad = True

class VisionEncoder(nn.Module):
    def __init__(self, num_unfrozen_layers=Config.NUM_UNFROZEN_LAYERS):
        super().__init__()
        self.model = SwinModel.from_pretrained(Config.VISION_MODEL)
        self.image_processor = AutoImageProcessor.from_pretrained(Config.VISION_MODEL)

        freeze_layers(self.model, num_unfrozen_layers)

    def forward(self, images):
        # Ensure images are in the correct format (e.g., torch.float32)
        if images.dtype != torch.float32:
            images = images.float()

        # Normalize the images using the image_processor's normalization parameters
        images = self.normalize_images(images)

        outputs = self.model(pixel_values=images)
        return outputs.last_hidden_state.mean(dim=1)

    def normalize_images(self, images):
        # Get the normalization parameters from the image_processor
        image_mean = torch.tensor(self.image_processor.image_mean).view(1, 3, 1, 1).to(images.device)
        image_std = torch.tensor(self.image_processor.image_std).view(1, 3, 1, 1).to(images.device)

        # Normalize the images
        normalized_images = (images - image_mean) / image_std
        return normalized_images