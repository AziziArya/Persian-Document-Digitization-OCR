import os
import sys
import csv
import random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import filter_ctc_feasible, build_batch

BASE_DIR = "."
MAX_WIDTH = 400
TARGET_HEIGHT = 48
MAX_LABEL_LEN = 60
DOWNSAMPLE_FACTOR = 4

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
N_VAL = int(sys.argv[2]) if len(sys.argv) > 2 else 128

rows = list(csv.DictReader(open(os.path.join(BASE_DIR, "data/processed/manifest.csv"), encoding="utf-8")))
kept = filter_ctc_feasible(rows, BASE_DIR, downsample_factor=DOWNSAMPLE_FACTOR, max_width=MAX_WIDTH)
random.seed(0)
random.shuffle(kept)
n_val_total = max(N_VAL, int(0.1 * len(kept)))
val_rows, train_rows = kept[:n_val_total], kept[n_val_total:]

train_rows = train_rows[:N_TRAIN]
val_rows = val_rows[:N_VAL]
print(f"caching {len(train_rows)} train + {len(val_rows)} val samples...")

x_tr, y_tr, inlens_tr, lbllens_tr = build_batch(train_rows, BASE_DIR, TARGET_HEIGHT, MAX_WIDTH, MAX_LABEL_LEN)
x_val, y_val, inlens_val, lbllens_val = build_batch(val_rows, BASE_DIR, TARGET_HEIGHT, MAX_WIDTH, MAX_LABEL_LEN)
true_texts_val = [r["label"] for r in val_rows]

os.makedirs("models", exist_ok=True)
np.savez(
    "models/cached_train_data.npz",
    x_tr=x_tr, y_tr=y_tr, inlens_tr=inlens_tr, lbllens_tr=lbllens_tr,
    x_val=x_val, y_val=y_val, inlens_val=inlens_val, lbllens_val=lbllens_val,
)
with open("models/cached_val_texts.txt", "w", encoding="utf-8") as f:
    for t in true_texts_val:
        f.write(t.replace("\n", " ") + "\n")

print("cached:", x_tr.shape, x_val.shape)
