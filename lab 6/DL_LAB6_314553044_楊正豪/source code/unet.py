import torch
import torch.nn as nn
from diffusers import UNet2DModel
NUM_OBJECTS = 24   
COND_DIM    = 512   


class ConditionEmbedding(nn.Module):
    def __init__(self, num_objects=NUM_OBJECTS, cond_dim=COND_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_objects, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )

    def forward(self, label):
        return self.net(label)


class Unet(nn.Module):
    def __init__(
        self,
        image_size   = 64,
        num_objects  = NUM_OBJECTS,
        cond_dim     = COND_DIM,
    ):
        super().__init__()

        self.cond_dim = cond_dim

        # condition embedding
        self.cond_emb = ConditionEmbedding(num_objects, cond_dim)

        # UNet 
        self.unet = UNet2DModel(
            sample_size        = image_size,
            in_channels        = 3,#3 + cond_dim,
            out_channels       = 3,
            class_embed_type   = "identity",
            layers_per_block   = 2,
            block_out_channels = (128, 256, 256, 512),
            down_block_types   = (
                "DownBlock2D",       # 64 → 32
                "DownBlock2D",       # 32 → 16
                "AttnDownBlock2D",   # 16 → 8
                "AttnDownBlock2D",   # 8  → 4
            ),
            up_block_types = (
                "AttnUpBlock2D",     # 4  → 8
                "AttnUpBlock2D",     # 8  → 16
                "UpBlock2D",         # 16 → 32
                "UpBlock2D",         # 32 → 64
            ),
        )

    def forward(self, x, timestep, label):
        cond = self.cond_emb(label)
        
        return self.unet(x, timestep, class_labels=cond).sample


