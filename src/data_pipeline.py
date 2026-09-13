"""
charset + پایپ‌لاین داده برای مدل CRNN+CTC.
"""
import string
import numpy as np
import cv2

# --- charset مشترک فارسی + انگلیسی + ارقام + علائم ---
PERSIAN_LETTERS = list("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
PERSIAN_EXTRA = list("ءآأؤئةيكھىـ")  # شکل‌های جایگزین رایج + تطویل
PERSIAN_DIGITS = list("۰۱۲۳۴۵۶۷۸۹")
LATIN_DIGITS = list(string.digits)
LATIN_LOWER = list(string.ascii_lowercase)
LATIN_UPPER = list(string.ascii_uppercase)
PUNCT = list(".,!?:;-()\"'/")
SPACE = [" "]

CHARS = sorted(set(
    PERSIAN_LETTERS + PERSIAN_EXTRA + PERSIAN_DIGITS +
    LATIN_DIGITS + LATIN_LOWER + LATIN_UPPER + PUNCT + SPACE
))

CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 برای CTC blank (آخرین ایندکس)
BLANK_IDX = len(CHARS)


def encode_label(text):
    """رشته متن را به لیست اندیس‌های کاراکتر تبدیل می‌کند. کاراکترهای ناشناس رد می‌شوند."""
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


def decode_indices(indices):
    return "".join(IDX_TO_CHAR.get(i, "") for i in indices)


def load_and_prepare_image(image_path, target_height=48, max_width=400):
    """عکس خط را می‌خواند، به ارتفاع ثابت نرمال و به max_width پد می‌کند."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    scale = target_height / h
    new_w = min(max_width, max(1, int(w * scale)))
    img = cv2.resize(img, (new_w, target_height))
    img = img.astype(np.float32) / 255.0
    # پد کردن به max_width با سفید (1.0) در سمت راست
    padded = np.ones((target_height, max_width), dtype=np.float32)
    padded[:, :new_w] = img
    return padded, new_w


def filter_ctc_feasible(rows, base_dir, downsample_factor=4, max_width=400, min_margin=1):
    """
    نمونه‌هایی که input_length مدل (بعد از downsample) کمتر از طول لیبل باشد را حذف می‌کند.
    CTC اصولاً امکان‌پذیر نیست اگر تعداد گام‌های زمانی از تعداد کاراکترهای لیبل کمتر باشد
    (هر کاراکتر خروجی حداقل به یک گام زمانی نیاز دارد، و برای کاراکترهای تکراری پشت سر هم
    حتی به یک blank میانی هم نیاز است) — این نمونه‌ها باید قبل از آموزش حذف شوند، وگرنه
    loss آن‌ها inf/nan می‌شود.
    """
    import os
    kept = []
    for r in rows:
        enc = encode_label(r["label"])
        if len(enc) == 0:
            continue
        path = os.path.join(base_dir, r["image_path"])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        scale = 48 / h
        real_w = min(max_width, max(1, int(w * scale)))
        input_len = max(1, int(real_w / downsample_factor))
        # قانون واقعی CTC: برای هر جفت کاراکتر تکراری پشت‌سرهم، یک گام زمانی اضافه
        # (برای blank میانی) لازم است، وگرنه هیچ alignment معتبری وجود ندارد
        # ("No valid path found") و loss آن نمونه inf می‌شود.
        repeats = sum(1 for i in range(1, len(enc)) if enc[i] == enc[i - 1])
        required_len = len(enc) + repeats
        if input_len >= required_len + min_margin:
            kept.append(r)
    return kept


def build_batch(rows, base_dir, target_height=48, max_width=400, max_label_len=60):
    """یک batch کامل (تصاویر + لیبل‌های پد شده + طول واقعی هرکدام) برای CTC می‌سازد."""
    import os
    imgs, labels, input_lens, label_lens = [], [], [], []
    for r in rows:
        path = os.path.join(base_dir, r["image_path"])
        result = load_and_prepare_image(path, target_height, max_width)
        if result is None:
            continue
        img, real_w = result
        enc = encode_label(r["label"])
        if len(enc) == 0 or len(enc) > max_label_len:
            continue
        imgs.append(img)
        # طول دنباله بعد از CNN (تقریبی؛ در نوت‌بوک واقعی از خروجی مدل محاسبه می‌شود)
        input_lens.append(real_w)
        padded_label = enc + [BLANK_IDX] * (max_label_len - len(enc))
        labels.append(padded_label)
        label_lens.append(len(enc))
    return (np.stack(imgs)[..., np.newaxis], np.array(labels),
            np.array(input_lens), np.array(label_lens))


if __name__ == "__main__":
    import csv
    rows = list(csv.DictReader(open("data/processed/manifest.csv", encoding="utf-8")))
    print("charset size:", len(CHARS), "| NUM_CLASSES (with blank):", NUM_CLASSES)
    sample = rows[:8]
    x, y, in_lens, lbl_lens = build_batch(sample, ".")
    print("images batch shape:", x.shape)
    print("labels batch shape:", y.shape)
    print("input_lens:", in_lens)
    print("label_lens:", lbl_lens)
    # round-trip چک: انکود و دیکود باید متن اصلی رو برگردونه
    for r in sample[:3]:
        enc = encode_label(r["label"])
        dec = decode_indices(enc)
        match = "✅" if dec == r["label"] else "❌"
        print(f"{match} original={r['label']!r}  decoded={dec!r}")
