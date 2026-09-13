"""
جداسازی خط از یک تصویر سند کامل (باینری‌شده و deskew شده)، با روش Projection Profile افقی.
"""
import numpy as np
import cv2


def segment_lines(bw, min_line_height=8, gap_threshold=2):
    """مرزهای عمودی (y_start, y_end) هر خط متن را در تصویر باینری برمی‌گرداند."""
    row_sums = np.sum(bw > 0, axis=1)
    is_text_row = row_sums > gap_threshold
    lines = []
    start = None
    for y, has_text in enumerate(is_text_row):
        if has_text and start is None:
            start = y
        elif not has_text and start is not None:
            if y - start >= min_line_height:
                lines.append((start, y))
            start = None
    if start is not None and len(is_text_row) - start >= min_line_height:
        lines.append((start, len(is_text_row)))
    return lines


def merge_close_boxes(boxes, max_gap=15):
    """
    خط‌های خیلی نزدیک به هم را با هم ادغام می‌کند.
    لازم است چون در فونت‌های دست‌نویس/نستعلیق، نقطه‌ها یا اجزای بالا/پایین حروف
    گاهی با یک فاصله‌ی خیلی کوچک از بدنه‌ی اصلی خط جدا می‌افتند و به‌اشتباه به‌عنوان
    یک خط مستقل تشخیص داده می‌شوند. فاصله‌ی واقعی بین دو خط معمولاً خیلی بزرگ‌تر از
    max_gap است، پس این ادغام خط‌های واقعی را با هم قاطی نمی‌کند.
    """
    if not boxes:
        return boxes
    merged = [list(boxes[0])]
    for y0, y1 in boxes[1:]:
        prev_y0, prev_y1 = merged[-1]
        if y0 - prev_y1 < max_gap:
            merged[-1][1] = y1  # ادغام با خط قبلی
        else:
            merged.append([y0, y1])
    return [tuple(b) for b in merged]


def crop_and_normalize(bw, box, target_height=48, pad=6):
    """یک خط را بر اساس مرزهای آن کراپ کرده، حاشیه اضافه را حذف و به ارتفاع ثابت نرمال می‌کند."""
    y0, y1 = box
    y0 = max(0, y0 - pad)
    y1 = min(bw.shape[0], y1 + pad)
    crop = bw[y0:y1, :]
    cols = np.where(np.sum(crop > 0, axis=0) > 0)[0]
    if len(cols) > 0:
        x0, x1 = max(0, cols[0] - pad), min(crop.shape[1], cols[-1] + pad)
        crop = crop[:, x0:x1]
    h, w = crop.shape
    if h == 0 or w == 0:
        return None
    scale = target_height / h
    new_w = max(1, int(w * scale))
    resized = cv2.resize(crop, (new_w, target_height))
    return resized


def segment_and_crop_lines(bw, target_height=48, min_line_height=8, gap_threshold=2,
                            pad=6, merge_gap=15):
    """تابع سطح‌بالا: جداسازی + ادغام قطعات نزدیک + کراپ + نرمال‌سازی همه خطوط یک سند."""
    boxes = segment_lines(bw, min_line_height=min_line_height, gap_threshold=gap_threshold)
    boxes = merge_close_boxes(boxes, max_gap=merge_gap)
    crops = []
    for box in boxes:
        c = crop_and_normalize(bw, box, target_height=target_height, pad=pad)
        if c is not None:
            crops.append(c)
    return crops, boxes
