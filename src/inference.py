"""
پایپ‌لاین کامل Inference: از یک عکس سند کامل تا متن نهایی.
پیش‌پردازش -> جداسازی خط -> تشخیص متن هر خط (CRNN+CTC) -> بازسازی سند -> JSON/CSV
"""
import os
import sys
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
import preprocessing
import segmentation
from model import build_crnn
from data_pipeline import NUM_CLASSES, BLANK_IDX, decode_indices
from decode import ctc_greedy_decode

TARGET_HEIGHT = 48
MAX_WIDTH = 400
DOWNSAMPLE_FACTOR = 4
VARIANT_KW = dict(lstm_units=384, extra_conv=True, dropout=0.2)  # باید با چک‌پوینت هماهنگ باشد


def load_ocr_model(weights_path, variant_kw=None):
    variant_kw = variant_kw if variant_kw is not None else VARIANT_KW
    model = build_crnn(TARGET_HEIGHT, MAX_WIDTH, NUM_CLASSES - 1, **variant_kw)
    model.load_weights(weights_path)
    return model


def _prepare_line_for_model(line_crop):
    """خروجی segment_and_crop_lines (باینری، متن=255) را برای مدل آماده می‌کند."""
    h, w = line_crop.shape
    if w > MAX_WIDTH:
        # فقط عرض را به MAX_WIDTH محدود کن، ارتفاع باید همیشه دقیقا TARGET_HEIGHT بماند
        line_crop = cv2.resize(line_crop, (MAX_WIDTH, TARGET_HEIGHT))
        w = MAX_WIDTH
    elif h != TARGET_HEIGHT:
        line_crop = cv2.resize(line_crop, (w, TARGET_HEIGHT))
    # مدل روی تصویر خاکستری عادی (متن تیره روی پس‌زمینه روشن، مقیاس 0..1) آموزش دیده
    inv = 255 - line_crop  # برگرداندن به «متن تیره روی سفید» مثل داده آموزشی
    normed = inv.astype(np.float32) / 255.0
    padded = np.ones((TARGET_HEIGHT, MAX_WIDTH), dtype=np.float32)
    padded[:, :w] = normed
    return padded, w


def run_ocr_on_document(image_path, model, save_json_path=None):
    """
    ورودی: مسیر عکس یک سند کامل (چند خط)
    خروجی: dict شامل متن هر خط + متن کامل بازسازی‌شده
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"تصویر خوانده نشد: {image_path}")

    bw, angle = preprocessing.preprocess_document(gray)
    crops, boxes = segmentation.segment_and_crop_lines(bw, target_height=TARGET_HEIGHT, merge_gap=15)

    if not crops:
        return {"image": image_path, "skew_angle": angle, "lines": [], "full_text": ""}

    batch = np.zeros((len(crops), TARGET_HEIGHT, MAX_WIDTH, 1), dtype=np.float32)
    real_widths = []
    for i, c in enumerate(crops):
        padded, w = _prepare_line_for_model(c)
        batch[i, :, :, 0] = padded
        real_widths.append(w)

    y_pred = model.predict(batch, verbose=0)
    T = y_pred.shape[1]
    real_input_lens = np.minimum(T, np.maximum(1, (np.array(real_widths) / DOWNSAMPLE_FACTOR).astype(np.int32)))
    decoded = ctc_greedy_decode(y_pred, real_input_lens, BLANK_IDX)
    texts = [decode_indices(d) for d in decoded]

    lines_out = []
    for (y0, y1), text in zip(boxes, texts):
        lines_out.append({"y_start": int(y0), "y_end": int(y1), "text": text})

    result = {
        "image": image_path,
        "skew_angle_deg": round(float(angle), 2),
        "num_lines": len(lines_out),
        "lines": lines_out,
        "full_text": "\n".join(l["text"] for l in lines_out),
    }

    if save_json_path:
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    import glob
    weights = "models/example_checkpoint_regularized.weights.h5"
    model = load_ocr_model(weights)
    test_images = ["samples/test_doc_fa_printed.png"]
    if not test_images:
        print("هیچ عکس تستی در samples/ پیدا نشد")
    else:
        for img_path in test_images[:1]:
            print(f"در حال پردازش: {img_path}")
            result = run_ocr_on_document(img_path, model)
            print(json.dumps(result, ensure_ascii=False, indent=2)[:1000])
