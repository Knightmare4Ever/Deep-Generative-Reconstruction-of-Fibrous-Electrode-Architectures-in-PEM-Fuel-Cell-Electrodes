import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class RefinerGenerator(nn.Module):
    def __init__(self, num_classes, z_dim):
        super().__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim

        self.enc1 = nn.Conv2d(num_classes, 64, 4, 2, 1)
        self.enc2 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128))
        self.enc3 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256))
        self.enc4 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512))

        self.bottleneck = nn.Sequential(nn.Conv2d(512 + z_dim, 512, 3, 1, 1), nn.ReLU())

        self.dec1 = self._up(512, 256)
        self.dec2 = self._up(256 * 2, 128)
        self.dec3 = self._up(128 * 2, 64)
        self.dec4 = self._up(64 * 2, 32)
        self.final = nn.Conv2d(32, num_classes, 3, 1, 1)

    def _up(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x, z):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        z_tiled = z.view(z.size(0), z.size(1), 1, 1).expand(-1, -1, e4.size(2), e4.size(3))
        b = torch.cat([e4, z_tiled], dim=1)
        b = self.bottleneck(b)

        d1 = self.dec1(b);
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.dec2(d1);
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.dec3(d2);
        d3 = torch.cat([d3, e1], dim=1)
        d4 = self.dec4(d3)
        return self.final(d4)


class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(num_classes, 64, 4, 2, 1)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(256, 1, 4, 1, 0))
        )

    def forward(self, x): return self.net(x)