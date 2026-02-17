import torch
import os

DATA_PATH = "data/ALL_CNN.tif"
OUTPUT_DIR = "final_clean_fast"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIBER_VAL = 6
BINDER_VAL = 4

EPOCHS = 350
BATCH_SIZE = 4
LR = 0.0002
Z_DIM = 64
LAMBDA_CE = 10.0
LAMBDA_ADV = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "final_model_checkpoint.pt")

GEN_COUNT = 200
WARMUP_STEPS = 50
BINDER_MAX_DISTANCE_FROM_FIBER = 25.0
FIBER_INFLUENCE_STRENGTH = 0.5
MACRO_THRESHOLD = 0.48
Z_INTENSITY = 0.75
PORE_THRESHOLD = 0.58
EVOLUTION_RATE = 0.3
DRAG = 0.05
WANDER_STRENGTH = 0.02
MACRO_MORPH_RATE = 0.08
PORE_MORPH_RATE = 0.15
Z_INTERP_FRAMES = 150