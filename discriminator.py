#discriminator
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def disc_block(in_c, out_c, stride, batch_norm=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            disc_block(in_channels, 64, stride=1, batch_norm=False),
            disc_block(64, 64, stride=2),
            disc_block(64, 128, stride=1),
            disc_block(128, 128, stride=2),
            disc_block(128, 256, stride=1),
            disc_block(256, 256, stride=2),
            disc_block(256, 512, stride=1),
            disc_block(512, 512, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.model(x)
