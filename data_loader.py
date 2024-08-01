import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import GPT2Tokenizer
from config import Config

class CoinDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_length=1024, tokenizer_path="./gpt2_finetuned_captioner"):
        self.data_dir = data_dir
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.image_caption_pairs = []
        self.transform = transform or transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.CenterCrop(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for img_name in os.listdir(data_dir):
            if img_name.endswith('.jpg'):
                txt_name = os.path.splitext(img_name)[0] + '.txt'
                txt_path = os.path.join(data_dir, txt_name)
                img_path = os.path.join(data_dir, img_name)
                
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    
                    # Truncate caption if it's too long
                    encoded_caption = self.tokenizer.encode(caption, max_length=self.max_length, truncation=True)
                    self.image_caption_pairs.append((img_path, self.tokenizer.decode(encoded_caption)))

        print(f"Total valid image-caption pairs: {len(self.image_caption_pairs)}")

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        img_path, caption = self.image_caption_pairs[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Add EOS token to caption
        caption = caption + self.tokenizer.eos_token
        caption = "Describe this coin: " + caption
        
        return image, caption

# The get_data_loaders function remains the same
def get_data_loaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    dataset = CoinDataset(Config.DATA_DIR, max_length=1024, tokenizer_path=Config.CAPTION_MODEL)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader