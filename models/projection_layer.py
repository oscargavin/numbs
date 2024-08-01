import torch.nn as nn

class ProjectionLayer(nn.Module):
    def __init__(self, vision_dim, language_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, language_dim * 2),
            nn.ReLU(),
            nn.Linear(language_dim * 2, language_dim)
        )
        self.layer_norm = nn.LayerNorm(language_dim)

    def forward(self, x):
        # Ensure input is 3D
        if x.dim() == 2:
            x = x.unsqueeze(1)
        projected = self.projection(x)
        return self.layer_norm(projected)