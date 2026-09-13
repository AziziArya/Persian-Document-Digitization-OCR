"""
مدل CRNN + CTC برای تشخیص متن خط (فارسی/انگلیسی، طول متغیر).
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def build_crnn(input_height=48, input_width=400, num_classes=126, dropout=0.0,
                lstm_units=256, extra_conv=False):
    """
    CNN (کاهش ارتفاع تا ۱) + BiLSTM x2 + Dense(num_classes+blank)
    خروجی: (batch, time_steps, num_classes)  -- آماده برای CTC loss

    سه حالت استفاده (مطابق الگوی همیشگی: Baseline / Improved / Regularized):
      - baseline    : build_crnn(...)                                  (dropout=0, extra_conv=False)
      - improved    : build_crnn(..., lstm_units=384, extra_conv=True) (ظرفیت بیشتر، بدون regularization)
      - regularized : build_crnn(..., lstm_units=384, extra_conv=True, dropout=0.2)
    """
    inp = layers.Input(shape=(input_height, input_width, 1), name="image")

    x = layers.Conv2D(64, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)                     # h/2, w/2
    if dropout: x = layers.Dropout(dropout * 0.5)(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)                     # h/4, w/4
    if dropout: x = layers.Dropout(dropout * 0.5)(x)

    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 1))(x)                     # h/8, w/4
    if dropout: x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 1))(x)                     # h/16, w/4
    if dropout: x = layers.Dropout(dropout)(x)

    if extra_conv:  # لایه‌ی اضافه برای نسخه Improved/Regularized (ظرفیت بیشتر)
        x = layers.Conv2D(512, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if dropout: x = layers.Dropout(dropout)(x)

    # ارتفاع را دقیقا به ۱ می‌رساند (48/16=3 -> با کرنل (3,1) به ۱ می‌رسد)
    x = layers.Conv2D(512, (3, 1), padding="valid", activation="relu")(x)

    # (batch, 1, w/4, 512) -> (batch, w/4, 512)
    x = layers.Reshape((-1, 512))(x)

    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    if dropout: x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    if dropout: x = layers.Dropout(dropout)(x)

    out = layers.Dense(num_classes + 1, activation="softmax", name="logits")(x)  # +1 = CTC blank

    name = "crnn_baseline"
    if extra_conv and not dropout:
        name = "crnn_improved"
    elif extra_conv and dropout:
        name = "crnn_regularized"
    model = models.Model(inp, out, name=name)
    return model


def ctc_loss_fn(y_true, y_pred, input_length, label_length):
    """y_true: (batch, max_label_len)  y_pred: (batch, T, num_classes+1)"""
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


if __name__ == "__main__":
    import sys
    import csv
    import numpy as np
    sys.path.insert(0, ".")
    from data_pipeline import build_batch, NUM_CLASSES

    model = build_crnn(input_height=48, input_width=400, num_classes=NUM_CLASSES - 1)
    model.summary()

    rows = list(csv.DictReader(open("data/processed/manifest.csv", encoding="utf-8")))
    x, y, in_lens_pixels, label_lens = build_batch(rows[:8], ".", max_width=400)
    print("input batch:", x.shape)

    y_pred = model.predict(x, verbose=0)
    print("model output shape:", y_pred.shape)  # (8, T, num_classes+1)

    T = y_pred.shape[1]
    downsample_factor = 400 / T
    print("time_steps:", T, "| downsample_factor:", downsample_factor)

    # input_length واقعی هر نمونه (بر اساس عرض واقعی قبل از پد، مقیاس‌شده به T)
    real_input_lens = np.minimum(T, np.maximum(1, (in_lens_pixels / downsample_factor).astype(np.int32)))
    print("real_input_lens:", real_input_lens)
    print("label_lens:", label_lens)

    # چک حیاتی CTC: input_length باید >= label_length باشد وگرنه loss نامعتبر می‌شود
    bad = np.sum(real_input_lens < label_lens)
    print("تعداد نمونه‌هایی که input_length < label_length دارند:", bad)

    loss = ctc_loss_fn(
        tf.constant(y, dtype=tf.int32),
        tf.constant(y_pred, dtype=tf.float32),
        tf.constant(real_input_lens.reshape(-1, 1), dtype=tf.int32),
        tf.constant(label_lens.reshape(-1, 1), dtype=tf.int32),
    )
    print("CTC loss per sample:", loss.numpy().flatten())
    print("mean CTC loss:", float(tf.reduce_mean(loss)))
