import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import config
from models import RefinerGenerator, Discriminator
from data_loader import load_and_preprocess_data
from engine import train_model, generate_stack
import torch


def main():
    # 1. Load Data
    gpu_data, val_to_idx, idx_to_val, num_classes, (H, W) = load_and_preprocess_data(
        config.DATA_PATH, config.DEVICE
    )

    # 2. Initialize Models
    netG = RefinerGenerator(num_classes, config.Z_DIM).to(config.DEVICE)
    netD = Discriminator(num_classes).to(config.DEVICE)

    optG = torch.optim.Adam(netG.parameters(), lr=config.LR, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=config.LR, betas=(0.5, 0.999))

    # 3. Train
    print("Starting Training...")
    train_model(gpu_data, netG, netD, optG, optD, num_classes, val_to_idx)

    # 4. Generate
    print("Starting Generation...")
    generate_stack(netG, num_classes, idx_to_val, val_to_idx[config.FIBER_VAL], val_to_idx[config.BINDER_VAL], gpu_data)


if __name__ == "__main__":
    main()