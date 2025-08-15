#generator
import torch
import torch.nn as nn

# Residual-in-Residual Dense Block (better than plain residual)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_residual_blocks=23):  # deeper
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.residuals = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])

        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.upsample = nn.Sequential(
            self._upsample_block(64),
            self._upsample_block(64)
        )

        self.final = nn.Conv2d(64, in_channels, kernel_size=9, stride=1, padding=4)

    def _upsample_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.mid(x) + initial
        x = self.upsample(x)
        return torch.tanh(self.final(x))

