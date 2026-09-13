"""
دیکود خروجی CTC (greedy) و محاسبه‌ی نرخ خطای کاراکتر (CER).
"""
import numpy as np
import tensorflow as tf


def ctc_greedy_decode(y_pred, input_lengths, blank_idx):
    """
    y_pred: (batch, T, num_classes+1) -- خروجی softmax مدل
    input_lengths: (batch,) طول واقعی هر نمونه (قبل از پد)
    خروجی: لیستی از رشته‌های اندیس (قبل از تبدیل به کاراکتر)، تکرارها و blank حذف شده
    """
    pred_indices = np.argmax(y_pred, axis=-1)  # (batch, T)
    results = []
    for i in range(pred_indices.shape[0]):
        seq = pred_indices[i][: input_lengths[i]]
        collapsed = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank_idx:
                collapsed.append(int(idx))
            prev = idx
        results.append(collapsed)
    return results


def levenshtein(a, b):
    """فاصله ویرایشی بین دو رشته/دنباله (برای محاسبه CER/WER)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[m]


def compute_cer(pred_texts, true_texts):
    """میانگین وزنی CER روی یک مجموعه (مجموع فاصله ویرایشی / مجموع طول متن واقعی)."""
    total_edits, total_chars = 0, 0
    for pred, true in zip(pred_texts, true_texts):
        total_edits += levenshtein(pred, true)
        total_chars += max(1, len(true))
    return total_edits / total_chars


if __name__ == "__main__":
    import sys
    import os
    import csv
    sys.path.insert(0, os.path.dirname(__file__))
    from data_pipeline import filter_ctc_feasible, build_batch, decode_indices, NUM_CLASSES, BLANK_IDX
    from model import build_crnn

    BASE_DIR = "."
    rows = list(csv.DictReader(open(os.path.join(BASE_DIR, "data/processed/manifest.csv"), encoding="utf-8")))
    kept = filter_ctc_feasible(rows, BASE_DIR, downsample_factor=4, max_width=400)

    import random
    random.seed(0)
    random.shuffle(kept)
    test_rows = kept[:16]

    base_model = build_crnn(48, 400, NUM_CLASSES - 1)
    base_model.load_weights("models/best_base_model.weights.h5")
    print("وزن‌های آموزش‌دیده (تست کوتاه قبلی) لود شد ✅")

    x, y, in_lens_pixels, label_lens = build_batch(test_rows, BASE_DIR, max_width=400, max_label_len=60)
    y_pred = base_model.predict(x, verbose=0)
    T = y_pred.shape[1]
    real_input_lens = np.minimum(T, np.maximum(1, (in_lens_pixels / 4).astype(np.int32)))

    decoded_indices = ctc_greedy_decode(y_pred, real_input_lens, BLANK_IDX)
    pred_texts = [decode_indices(d) for d in decoded_indices]
    true_texts = [r["label"] for r in test_rows]

    for p, t in zip(pred_texts[:6], true_texts[:6]):
        print(f"واقعی : {t!r}")
        print(f"پیش‌بینی: {p!r}")
        print("---")

    cer = compute_cer(pred_texts, true_texts)
    print(f"CER روی این batch کوچک (بعد از فقط 3 epoch تست): {cer:.3f}")
    print("(طبیعیه که هنوز دقتی نداره -- این فقط تست درستی کد دیکود/CER است، نه مدل نهایی)")
