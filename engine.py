import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import numpy as np
import os
import tifffile as tiff
from scipy.ndimage import label, distance_transform_edt
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from utils import to_onehot, get_gaussian_kernel, generate_foam_noise, smooth_mask
import config

import config
from utils import generate_foam_noise, get_gaussian_kernel, to_onehot


def train_model(gpu_data, netG, netD, optG, optD, num_classes, val_to_idx):
    criterion_ce = torch.nn.CrossEntropyLoss()
    g_kernel = get_gaussian_kernel(5, 2.0, device=config.DEVICE)
    h, w = gpu_data.shape[1], gpu_data.shape[2]

    for epoch in tqdm(range(config.EPOCHS), desc="Training"):
        indices = torch.randperm(len(gpu_data))
        for i in range(0, len(indices), config.BATCH_SIZE):
            batch_idx = indices[i:i + config.BATCH_SIZE]
            if len(batch_idx) < config.BATCH_SIZE: continue

            real_idx = gpu_data[batch_idx]
            # Data Augmentation
            if random.random() > 0.5: real_idx = torch.flip(real_idx, [2])
            if random.random() > 0.5: real_idx = torch.flip(real_idx, [1])

            target_oh = to_onehot(real_idx, num_classes).to(config.DEVICE)

            # Mask generation logic
            fiber_mask = (real_idx == val_to_idx[config.FIBER_VAL])
            binder_mask = (real_idx == val_to_idx[config.BINDER_VAL])
            bg_mask = (~fiber_mask) & (~binder_mask)

            noise = generate_foam_noise((config.BATCH_SIZE, h, w), device=config.DEVICE)
            pore_mask = (noise > random.uniform(0.35, 0.65))
            noisy_binder = smooth_mask(binder_mask & pore_mask, g_kernel)

            # ... Rest of your input_idx logic ...
            input_idx = torch.zeros_like(real_idx)
            input_idx[bg_mask] = real_idx[bg_mask]
            input_idx[noisy_binder] = val_to_idx[config.BINDER_VAL]

            sx, sy = random.randint(-2, 2), random.randint(-2, 2)
            shifted_fibers = torch.roll(fiber_mask, shifts=(sy, sx), dims=(1, 2))
            input_idx[shifted_fibers] = val_to_idx[config.FIBER_VAL]

            input_oh = to_onehot(input_idx, num_classes).to(config.DEVICE)

            # --- D step ---
            optD.zero_grad()
            d_real = netD(target_oh)
            z = torch.randn(config.BATCH_SIZE, config.Z_DIM, device=config.DEVICE)
            fake_logits = netG(input_oh, z)
            fake_soft = F.softmax(fake_logits, dim=1)
            d_fake = netD(fake_soft.detach())
            loss_d = torch.mean(F.relu(1.0 - d_real)) + torch.mean(F.relu(1.0 + d_fake))
            loss_d.backward()
            optD.step()

            # --- G step ---
            optG.zero_grad()
            d_fake_g = netD(fake_soft)
            loss_g = -torch.mean(d_fake_g) * config.LAMBDA_ADV + criterion_ce(fake_logits, real_idx) * config.LAMBDA_CE
            loss_g.backward()
            optG.step()

    torch.save({'generator_state_dict': netG.state_dict()}, config.CHECKPOINT_PATH)


def generate_stack(netG, num_classes, idx_to_val, fiber_idx, binder_idx, gpu_data):
    netG.eval()
    H, W = gpu_data.shape[1], gpu_data.shape[2]
    device = config.DEVICE

    # 1. Setup Kernels
    final_smoothing_kernel = get_gaussian_kernel(5, 1.5, device=device)
    influence_blur_kernel = get_gaussian_kernel(31, 10.0, device=device)

    # 2. Initialize Fiber Physics
    seed_slice = gpu_data[0].cpu().numpy()
    fiber_mask_np = (seed_slice == fiber_idx).astype(int)
    labeled_fibers, num_features = label(fiber_mask_np)

    fiber_physics = []
    for fid in range(1, num_features + 1):
        mask_single = (labeled_fibers == fid).astype(int)
        props = regionprops(mask_single)
        if not props: continue
        angle = props[0].orientation
        dy, dx = -np.cos(angle), -np.sin(angle)
        mag = np.sqrt(dy ** 2 + dx ** 2)
        if mag > 0: dy /= mag; dx /= mag
        if random.random() > 0.5: dy, dx = -dy, -dx
        speed = random.uniform(12.0, 18.0)
        fiber_physics.append({
            "mask": torch.from_numpy(mask_single).to(device),
            "vel_y": dy * speed, "vel_x": dx * speed,
            "pos_y": 0.0, "pos_x": 0.0
        })

    # 3. Initialize Noise and Vectors
    macro_A = generate_foam_noise((1, H * 2, W * 2), [32, 16], [0.7, 0.3], device=device)
    macro_B = generate_foam_noise((1, H * 2, W * 2), [24, 12], [0.7, 0.3], device=device)
    pore_A = generate_foam_noise((1, H * 2, W * 2), [12, 6], [0.5, 0.5], device=device)
    pore_B = generate_foam_noise((1, H * 2, W * 2), [10, 5], [0.5, 0.5], device=device)

    pos_mA, pos_mB = [0.0, 0.0], [100.0, 100.0]
    pos_pA, pos_pB = [200.0, 200.0], [300.0, 300.0]
    velocity = torch.tensor([0.0, 0.0], device=device)
    acceleration = torch.tensor([0.0, 0.0], device=device)

    # 4. Latent Space (Z) Interpolation
    z_total_frames = config.WARMUP_STEPS + config.GEN_COUNT
    z_points = [torch.randn(1, config.Z_DIM, device=device) for _ in
                range(z_total_frames // config.Z_INTERP_FRAMES + 2)]
    z_list = [torch.lerp(z_points[i], z_points[i + 1], step / config.Z_INTERP_FRAMES)
              for i in range(len(z_points) - 1) for step in range(config.Z_INTERP_FRAMES)]

    # 5. Warm-up and Main Loop Setup
    initial_bg = gpu_data[0].clone()
    initial_bg[initial_bg > 0] = 0
    macro_strength_map = torch.zeros((H, W), device=device)
    pore_strength_map = torch.zeros((H, W), device=device)
    generated_stack = []

    # Combine Warm-up and Generation for efficiency
    print(f"--- Running {config.WARMUP_STEPS} warm-up and {config.GEN_COUNT} gen frames ---")

    with torch.no_grad():
        for i in tqdm(range(z_total_frames), desc="Total Evolution"):
            # Update Physics
            wander = torch.tensor([random.uniform(-1, 1), random.uniform(-1, 1)], device=device)
            acceleration = torch.clamp(acceleration + wander * config.WANDER_STRENGTH, -0.1, 0.1)
            velocity = (velocity + acceleration) * (1.0 - config.DRAG)

            pos_mA[0] += velocity[0];
            pos_mA[1] += velocity[1]
            pos_mB[0] -= velocity[0] * config.EVOLUTION_RATE;
            pos_mB[1] -= velocity[1] * config.EVOLUTION_RATE
            pos_pA[0] += velocity[0] * (1 + config.EVOLUTION_RATE);
            pos_pA[1] -= velocity[1] * (1 + config.EVOLUTION_RATE)
            pos_pB[0] -= velocity[0] * (config.EVOLUTION_RATE / 2);
            pos_pB[1] += velocity[1] * (config.EVOLUTION_RATE / 2)

            # Sample Noise
            syA, sxA = int(pos_mA[0]) % H, int(pos_mA[1]) % W
            sample_mA = macro_A[syA:syA + H, sxA:sxA + W]
            evolving_macro_noise = (sample_mA + macro_B[
                int(pos_mB[0]) % H: int(pos_mB[0]) % H + H, int(pos_mB[1]) % W: int(pos_mB[1]) % W + W]) / 2
            evolving_pore_noise = (pore_A[int(pos_pA[0]) % H: int(pos_pA[0]) % H + H, int(pos_pA[1]) % W: int(
                pos_pA[1]) % W + W] +
                                   pore_B[int(pos_pB[0]) % H: int(pos_pB[0]) % H + H, int(pos_pB[1]) % W: int(
                                       pos_pB[1]) % W + W]) / 2

            # Fiber Movement
            combined_fiber_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
            for f in fiber_physics:
                f["pos_y"] += f["vel_y"];
                f["pos_x"] += f["vel_x"]
                shifted = torch.roll(f["mask"], shifts=(int(round(f["pos_y"])), int(round(f["pos_x"]))), dims=(0, 1))
                combined_fiber_mask |= shifted.bool()

            # Gated Interaction Logic
            fiber_mask_float = combined_fiber_mask.float().unsqueeze(0).unsqueeze(0)
            fiber_influence = F.conv2d(fiber_mask_float, influence_blur_kernel, padding=15).squeeze()
            influenced_macro = (evolving_macro_noise + fiber_influence * config.FIBER_INFLUENCE_STRENGTH).clamp(0, 1)

            dist_map = distance_transform_edt(~combined_fiber_mask.cpu().numpy())
            permission = torch.from_numpy(dist_map <= config.BINDER_MAX_DISTANCE_FROM_FIBER).to(device)

            target_macro = (influenced_macro > config.MACRO_THRESHOLD).float() * permission.float()
            target_pore = (evolving_pore_noise > config.PORE_THRESHOLD).float()

            macro_strength_map = torch.lerp(macro_strength_map, target_macro, config.MACRO_MORPH_RATE)
            pore_strength_map = torch.lerp(pore_strength_map, target_pore, config.PORE_MORPH_RATE)

            # Only Generate/Refine after Warm-up
            if i >= config.WARMUP_STEPS:
                is_binder = (macro_strength_map > 0.5) & ~(pore_strength_map > 0.5)
                proposal = initial_bg.clone()
                proposal[is_binder] = binder_idx
                proposal[combined_fiber_mask] = fiber_idx

                # GAN Refinement
                prop_oh = to_onehot(proposal.unsqueeze(0), num_classes).to(device)
                z = z_list[i] * config.Z_INTENSITY
                refined_idx = torch.argmax(netG(prop_oh, z), dim=1).squeeze()

                # Post-Processing
                frame_np = refined_idx.cpu().numpy().astype(np.uint8)
                b_mask = torch.from_numpy(frame_np == binder_idx).float().view(1, 1, H, W).to(device)
                smoothed_b = (F.conv2d(b_mask, final_smoothing_kernel, padding=2) > 0.5).squeeze().cpu().numpy()
                cleaned_b = remove_small_objects(smoothed_b, min_size=60)

                final_f = frame_np.copy()
                final_f[final_f == binder_idx] = 0
                final_f[cleaned_b] = binder_idx
                generated_stack.append(final_f)

    # 6. Final Mapping and Saving
    final_indices_vol = np.array(generated_stack)
    final_mapped_vol = np.zeros_like(final_indices_vol)
    for index, value in idx_to_val.items():
        final_mapped_vol[final_indices_vol == index] = value

    out_path = os.path.join(config.OUTPUT_DIR, "generated_output.tif")
    tiff.imwrite(out_path, final_mapped_vol)
    print(f"Saved: {out_path}")

    plt.imshow(final_mapped_vol[len(final_mapped_vol) // 2], cmap='gray')
    plt.show()