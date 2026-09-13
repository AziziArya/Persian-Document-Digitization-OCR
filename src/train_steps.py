import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.insert(0, os.path.dirname(__file__))
from model import build_crnn
from data_pipeline import NUM_CLASSES, BLANK_IDX, decode_indices
from decode import ctc_greedy_decode, compute_cer

MAX_WIDTH = 400
TARGET_HEIGHT = 48
MAX_LABEL_LEN = 60
DOWNSAMPLE_FACTOR = 4
VARIANT_KW = dict(lstm_units=384, extra_conv=True, dropout=0.2)
BATCH_SIZE = 16

LATEST_W = "models/latest_regularized.weights.h5"
BEST_W = "models/best_regularized.weights.h5"
STATE_F = "models/train_state.json"
HIST_F = "models/train_history_regularized.json"

N_BATCHES_THIS_RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def build_training_model(base_model):
    labels = layers.Input(name="labels", shape=(MAX_LABEL_LEN,), dtype="float32")
    input_length = layers.Input(name="input_length", shape=(1,), dtype="int64")
    label_length = layers.Input(name="label_length", shape=(1,), dtype="int64")
    y_pred = base_model.output
    loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length])
    return models.Model([base_model.input, labels, input_length, label_length], loss_out)


data = np.load("models/cached_train_data.npz")
x_tr, y_tr, inlens_tr, lbllens_tr = data["x_tr"], data["y_tr"], data["inlens_tr"], data["lbllens_tr"]
x_val, y_val, inlens_val, lbllens_val = data["x_val"], data["y_val"], data["inlens_val"], data["lbllens_val"]
true_texts_val = [l.strip() for l in open("models/cached_val_texts.txt", encoding="utf-8")]

real_inlens_tr = np.minimum(MAX_WIDTH // DOWNSAMPLE_FACTOR,
                             np.maximum(1, (inlens_tr / DOWNSAMPLE_FACTOR).astype(np.int64))).reshape(-1, 1)
lbllens_tr_r = lbllens_tr.reshape(-1, 1).astype(np.int64)
real_inlens_val = np.minimum(MAX_WIDTH // DOWNSAMPLE_FACTOR,
                              np.maximum(1, (inlens_val / DOWNSAMPLE_FACTOR).astype(np.int64))).reshape(-1, 1)
lbllens_val_r = lbllens_val.reshape(-1, 1).astype(np.int64)

n_train = len(x_tr)
n_batches_per_epoch = n_train // BATCH_SIZE

base_model = build_crnn(TARGET_HEIGHT, MAX_WIDTH, NUM_CLASSES - 1, **VARIANT_KW)
training_model = build_training_model(base_model)
opt = tf.keras.optimizers.Adam(3e-4)
training_model.compile(optimizer=opt, loss={"ctc": lambda yt, yp: yp})

state = json.load(open(STATE_F)) if os.path.isfile(STATE_F) else {
    "global_step": 0, "epoch": 0, "best_val_loss": None, "order": None
}
history = json.load(open(HIST_F)) if os.path.isfile(HIST_F) else {"train_loss": [], "val_loss": [], "val_cer": []}

if os.path.isfile(LATEST_W):
    base_model.load_weights(LATEST_W)
    print(f"[resume] از global_step={state['global_step']} (epoch {state['epoch']}) ادامه می‌دهیم")
else:
    print("[start] شروع از صفر")

rng = np.random.RandomState(123 + state["epoch"])
order = state["order"] if state["order"] is not None else rng.permutation(n_train).tolist()

losses_this_run = []
pos_in_epoch = state["global_step"] % n_batches_per_epoch

for b in range(N_BATCHES_THIS_RUN):
    if pos_in_epoch >= n_batches_per_epoch:
        # یک epoch کامل شد -> شافل مجدد و ارزیابی روی validation
        pos_in_epoch = 0
        state["epoch"] += 1
        rng = np.random.RandomState(123 + state["epoch"])
        order = rng.permutation(n_train).tolist()

        val_loss = training_model.evaluate(
            {"image": x_val, "labels": y_val.astype(np.float32),
             "input_length": real_inlens_val, "label_length": lbllens_val_r},
            np.zeros((len(x_val), 1), dtype=np.float32), verbose=0)
        y_pred_val = base_model.predict(x_val, verbose=0)
        decoded = ctc_greedy_decode(y_pred_val, real_inlens_val.flatten(), BLANK_IDX)
        pred_texts = [decode_indices(d) for d in decoded]
        cer = compute_cer(pred_texts, true_texts_val)
        history["val_loss"].append(float(val_loss))
        history["val_cer"].append(float(cer))
        n_nonempty = sum(1 for p in pred_texts if p.strip())
        print(f"=== پایان epoch {state['epoch']}: val_loss={val_loss:.2f}  val_CER={cer:.3f}  non_empty={n_nonempty}/{len(pred_texts)} ===")

        base_model.save_weights(LATEST_W)
        if state["best_val_loss"] is None or val_loss < state["best_val_loss"]:
            state["best_val_loss"] = float(val_loss)
            base_model.save_weights(BEST_W)
            print(f"  -> بهترین مدل جدید ذخیره شد (val_loss={val_loss:.2f})")

    idx = order[pos_in_epoch * BATCH_SIZE:(pos_in_epoch + 1) * BATCH_SIZE]
    if len(idx) == 0:
        pos_in_epoch = n_batches_per_epoch  # force epoch rollover next iter
        continue
    batch_x = x_tr[idx]
    batch_y = y_tr[idx].astype(np.float32)
    batch_inlen = real_inlens_tr[idx]
    batch_lbllen = lbllens_tr_r[idx]
    loss = training_model.train_on_batch(
        {"image": batch_x, "labels": batch_y, "input_length": batch_inlen, "label_length": batch_lbllen},
        np.zeros((len(idx), 1), dtype=np.float32)
    )
    losses_this_run.append(float(loss))
    pos_in_epoch += 1
    state["global_step"] += 1

state["order"] = order
base_model.save_weights(LATEST_W)
history["train_loss"] += losses_this_run
json.dump(state, open(STATE_F, "w"))
json.dump(history, open(HIST_F, "w"))

print(f"\nاین اجرا: {len(losses_this_run)} batch, میانگین loss={np.mean(losses_this_run):.2f}")
print(f"global_step کل: {state['global_step']}  (epoch {state['epoch']})")
if history["val_cer"]:
    print("روند val_CER:", [round(c, 3) for c in history["val_cer"]])
