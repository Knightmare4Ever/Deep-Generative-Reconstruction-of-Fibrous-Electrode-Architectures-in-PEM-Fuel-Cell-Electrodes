import numpy as np
import tifffile as tiff
import torch


def load_and_preprocess_data(path, device):
    full_stack = tiff.imread(path)
    if full_stack.ndim == 2: full_stack = full_stack[np.newaxis, ...]

    train_stack = full_stack[80:150]
    h, w = train_stack.shape[1], train_stack.shape[2]

    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    if ph > 0 or pw > 0:
        train_stack = np.pad(train_stack, ((0, 0), (0, ph), (0, pw)), mode='constant', constant_values=0)

    unique_vals = sorted(np.unique(train_stack))
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    idx_to_val = {i: v for i, v in enumerate(unique_vals)}

    train_indices = np.vectorize(val_to_idx.get)(train_stack)
    gpu_data = torch.from_numpy(train_indices).long().to(device)

    return gpu_data, val_to_idx, idx_to_val, len(unique_vals), (h + ph, w + pw)