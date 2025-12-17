import torch
import torch.nn as nn
import os


class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 6 * 8, latent_dim), # input_image = (64*96)
            nn.Tanh() # keeps [-1, 1], good for PPO
        )

    def forward(self, x):
        return self.net(x)


_ENCODER_MODEL = None

def load_encoder(device):
    global _ENCODER_MODEL
    if _ENCODER_MODEL is None:
        model_path = os.path.join(os.path.dirname(__file__), "../agents/vision_encoder.pt")
        print(f"Loading Vision Encoder from: {model_path}")

        _ENCODER_MODEL = VAEEncoder().to(device)

        if os.path.exists(model_path):
            _ENCODER_MODEL.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("WARNING: Encoder weights not found! Using random weights.")

        _ENCODER_MODEL.eval()
        for param in _ENCODER_MODEL.parameters():
            param.requires_grad = False

    return _ENCODER_MODEL