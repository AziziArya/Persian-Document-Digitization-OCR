"""
توابع کمکی مشترک بین نوت‌بوک‌ها.
"""
import os


def find_project_root(start=None, marker="fonts"):
    """
    ریشه پروژه را مستقل از این‌که نوت‌بوک از کجا اجرا شده پیدا می‌کند
    (چه از ریشه پروژه، چه از داخل پوشه notebooks/).
    به‌جای os.getcwd() ساده که وابسته به محل اجراست، رو به بالا دنبال پوشه‌ی
    نشانه (پیش‌فرض: 'fonts') می‌گردد.
    """
    d = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.isdir(os.path.join(d, marker)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(
                f"ریشه پروژه (حاوی پوشه '{marker}') پیدا نشد. "
                f"مطمئن شوید نوت‌بوک داخل ساختار ریپازیتوری اجرا می‌شود."
            )
        d = parent
