"""
پیش‌پردازش تصویر برای پایپ‌لاین OCR فارسی/انگلیسی.
شامل: باینری‌سازی و اصلاح کجی (deskew).
"""
import numpy as np
import cv2


def binarize(gray_arr):
    """تصویر خاکستری را می‌گیرد و تصویر باینری برمی‌گرداند (پس‌زمینه=0، متن=255)."""
    blurred = cv2.GaussianBlur(gray_arr, (3, 3), 0)
    _, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return bw


def deskew(bw):
    """زاویه کجی سند را با minAreaRect پیدا و اصلاح می‌کند. ورودی/خروجی تصویر باینری است."""
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 20:
        return bw, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = bw.shape
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated, angle


def preprocess_document(gray_arr):
    """پایپ‌لاین کامل پیش‌پردازش: باینری‌سازی + deskew. یک تصویر خاکستری خام می‌گیرد."""
    bw = binarize(gray_arr)
    bw_deskewed, angle = deskew(bw)
    return bw_deskewed, angle
