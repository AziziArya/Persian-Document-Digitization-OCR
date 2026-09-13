"""
دموی گرافیکی (tkinter) برای سیستم OCR فارسی/انگلیسی.
دقیقاً به سبک دموی پروژه ارقام دست‌نویس: بارگذاری عکس -> پردازش -> نمایش متن + عکس نتیجه.
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
import cv2

from inference import load_ocr_model, run_ocr_on_document

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "models",
                             "example_checkpoint_regularized.weights.h5")

print("در حال بارگذاری مدل... (چند ثانیه طول می‌کشد)")
model = load_ocr_model(WEIGHTS_PATH)
print("مدل بارگذاری شد ✅")


def open_file_dialog():
    file_types = [('All image files', '*.png;*.jpg;*.jpeg;*.bmp'),
                  ('PNG Files', '*.png'), ('JPEG Files', '*.jpg'), ('Bitmap Files', '*.bmp')]
    file_path = filedialog.askopenfilename(filetypes=file_types)
    if not file_path:
        return

    lbl_status.config(text="در حال پردازش...")
    root.update()

    result = run_ocr_on_document(file_path, model)

    txt_output.delete("1.0", tk.END)
    txt_output.insert(tk.END, result["full_text"])
    lbl_status.config(text=f"{result['num_lines']} خط پیدا و تشخیص داده شد "
                            f"(زاویه اصلاح‌شده: {result['skew_angle_deg']}°)")

    # نمایش تصویر ورودی
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    pil_img.thumbnail((500, 500))
    tk_img = ImageTk.PhotoImage(pil_img)
    lbl_img.config(image=tk_img)
    lbl_img.image = tk_img

    # ذخیره خروجی JSON کنار عکس ورودی
    json_path = os.path.splitext(file_path)[0] + "_ocr_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("خروجی JSON ذخیره شد:", json_path)


root = tk.Tk()
root.title("OCR فارسی و انگلیسی — کلمات و جملات")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width, window_height = 900, 650
x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

btn_load = tk.Button(root, text="بارگذاری تصویر سند", command=open_file_dialog, font=("Tahoma", 12))
btn_load.pack(anchor="center", pady=10)

lbl_status = tk.Label(root, text="یک تصویر سند انتخاب کنید", font=("Tahoma", 11))
lbl_status.pack(anchor="center", pady=5)

frame = tk.Frame(root)
frame.pack(fill="both", expand=True, padx=10, pady=10)

lbl_img = tk.Label(frame)
lbl_img.pack(side="left", padx=10)

txt_output = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Tahoma", 12), width=40)
txt_output.pack(side="right", fill="both", expand=True, padx=10)

root.mainloop()
