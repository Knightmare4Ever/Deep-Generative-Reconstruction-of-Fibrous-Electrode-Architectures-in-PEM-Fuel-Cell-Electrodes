import torch
import torch.nn.functional as F
import math
import random

def to_onehot(indices, num_classes):
    return F.one_hot(indices, num_classes=num_classes).permute(0, 3, 1, 2).float()

def get_gaussian_kernel(kernel_size=5, sigma=2.0, device='cpu', channels=1):
    k = torch.tensor([math.exp(-x**2/(2*sigma**2)) for x in range(-(kernel_size//2), kernel_size//2+1)], device=device)
    k = k[:, None] * k[None, :]
    k /= k.sum()
    return k.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

def smooth_mask(mask, kernel):
    if mask.ndim == 2: mask = mask.unsqueeze(0).unsqueeze(0)
    if mask.ndim == 3: mask = mask.unsqueeze(1)
    x = mask.float()
    x = F.conv2d(x, kernel, padding=kernel.shape[2]//2)
    return (x > 0.5).squeeze()

def generate_foam_noise(shape, scales=[16, 8, 4], weights=[0.5, 0.25, 0.25], device='cpu'):
    B, H, W = shape
    combined_noise = torch.zeros(B, 1, H, W, device=device)
    for scale, weight in zip(scales, weights):
        noise = torch.rand(B, 1, H // scale, W // scale, device=device)
        upscaled_noise = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)
        combined_noise += upscaled_noise * weight
    combined_noise -= combined_noise.min()
    combined_noise /= combined_noise.max()
    return combined_noise.squeeze()