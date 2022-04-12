import torch
from torch import nn


class AtariNet(nn.Module):
    def __init__(self, bottleneck):
        super().__init__()

        self.bottleneck = bottleneck

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 128, (4, 4), stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, self.bottleneck),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.bottleneck, 256, (4, 4), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, (4, 4), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (4, 4), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, (2, 2), stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction

    def encode(self, x):
        embedding = self.encoder(x)
        return embedding

    def decode(self, embedding):
        reconstruction = self.decoder(embedding.view(-1, self.bottleneck, 1, 1))
        return reconstruction


if __name__ == "__main__":
    import gym
    import numpy as np
    from PIL import Image
    import cv2

    from torchvision.transforms.functional import resize, rgb_to_grayscale

    env = gym.make("MontezumaRevenge-v0")
    env.reset()
    frames = np.array([env.step(env.action_space.sample())[0] for _ in range(32)])
    frames = np.moveaxis(frames, 3, 1)  # (N x H x W x 3) to (N x 3 x H x W)
    frames = torch.from_numpy(frames).to(torch.float32)
    # Convert to grayscale
    frames = rgb_to_grayscale(frames)  # (N x 3 x H x W) to (N x 1 x H x W)
    # Resize
    frames = (resize(frames, (84, 84)) - 128) / 255  # (N x 1 x NEW_W x NEW_H)

    # frames = frames.detach().cpu().numpy().astype(np.uint8)
    # im = Image.fromarray(frames[0][0])

    net = AtariNet(32)
    x = net.encode(frames)
    print(x.shape)
    x = net.decode(x)
    print(x.shape)
    frames = (x * 255).detach().cpu().numpy().astype(np.uint8)
    im = Image.fromarray(frames[0][0])
    im.show()
