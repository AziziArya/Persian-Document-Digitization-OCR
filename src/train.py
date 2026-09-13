"""
آموزش مدل CRNN+CTC با الگوی استاندارد Keras (لایه CTC به‌عنوان بخشی از مدل آموزش).
"""
import os
import sys
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.insert(0, os.path.dirname(__file__))
from model import build_crnn
from data_pipeline import filter_ctc_feasible, build_batch, NUM_CLASSES, BLANK_IDX

MAX_WIDTH = 400
TARGET_HEIGHT = 48
MAX_LABEL_LEN = 60
DOWNSAMPLE_FACTOR = 4


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def build_training_model(base_model, num_classes):
    labels = layers.Input(name="labels", shape=(MAX_LABEL_LEN,), dtype="float32")
    input_length = layers.Input(name="input_length", shape=(1,), dtype="int64")
    label_length = layers.Input(name="label_length", shape=(1,), dtype="int64")

    y_pred = base_model.output
    loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )
    training_model = models.Model(
        inputs=[base_model.input, labels, input_length, label_length], outputs=loss_out
    )
    return training_model


def make_dataset(rows, base_dir, batch_size=32, shuffle=True):
    x, y, in_lens_pixels, label_lens = build_batch(
        rows, base_dir, target_height=TARGET_HEIGHT, max_width=MAX_WIDTH, max_label_len=MAX_LABEL_LEN
    )
    real_input_lens = np.minimum(
        MAX_WIDTH // DOWNSAMPLE_FACTOR,
        np.maximum(1, (in_lens_pixels / DOWNSAMPLE_FACTOR).astype(np.int64)),
    ).reshape(-1, 1)
    label_lens_r = label_lens.reshape(-1, 1).astype(np.int64)
    y = y.astype(np.float32)

    n = len(x)
    dummy_out = np.zeros((n, 1), dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices(
        ({"image": x, "labels": y, "input_length": real_input_lens, "label_length": label_lens_r}, dummy_out)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(2000, n))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, n


if __name__ == "__main__":
    BASE_DIR = "."
    rows = list(csv.DictReader(open(os.path.join(BASE_DIR, "data/processed/manifest.csv"), encoding="utf-8")))
    print("raw rows:", len(rows))

    kept = filter_ctc_feasible(rows, BASE_DIR, downsample_factor=DOWNSAMPLE_FACTOR, max_width=MAX_WIDTH)
    print("kept after CTC-feasibility filter:", len(kept))

    import random
    random.seed(0)
    random.shuffle(kept)
    n_val = int(0.1 * len(kept))
    val_rows, train_rows = kept[:n_val], kept[n_val:]
    print("train:", len(train_rows), "val:", len(val_rows))

    # فقط برای تست سریع پایپ‌لاین آموزش، یک زیرمجموعه خیلی کوچک با تعداد epoch کم
    train_rows_small = train_rows[:64]
    val_rows_small = val_rows[:16]

    train_ds, n_train = make_dataset(train_rows_small, BASE_DIR, batch_size=16)
    val_ds, n_val2 = make_dataset(val_rows_small, BASE_DIR, batch_size=16, shuffle=False)

    base_model = build_crnn(TARGET_HEIGHT, MAX_WIDTH, NUM_CLASSES - 1)
    training_model = build_training_model(base_model, NUM_CLASSES)
    training_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss={"ctc": lambda yt, yp: yp})

    os.makedirs("models", exist_ok=True)
    cb = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/best_base_model.weights.h5", monitor="val_loss",
            save_best_only=True, save_weights_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    history = training_model.fit(train_ds, validation_data=val_ds, epochs=3, callbacks=cb, verbose=2)
    print("final train loss:", history.history["loss"][-1])
    print("final val loss:", history.history["val_loss"][-1])
    print("loss history:", history.history["loss"])
