"""
runtime.py
==========
Tiện ích theo môi trường chạy (local / Kaggle).

Trên Kaggle notebook chạy qua papermill: output quá nhiều (thanh tiến trình theo batch, widget tqdm,
cảnh báo DataLoader) làm cell bị ngắt với "Timeout waiting for IOPub output". Vì vậy:
  * progress(): thanh tiến trình theo batch, mặc định TẮT trên Kaggle (config progress_bar: auto)
  * loader_kwargs(): giữ worker DataLoader sống qua các epoch (bớt tạo/huỷ worker + cảnh báo)
"""

from pathlib import Path

from tqdm.auto import tqdm

from vnlp import config


def on_kaggle() -> bool:
    return Path("/kaggle/input").exists()


def progress_enabled() -> bool:
    value = config.CFG.get("progress_bar", "auto")
    return not on_kaggle() if value == "auto" else bool(value)


def progress(iterable, **kwargs):
    return tqdm(iterable, disable=not progress_enabled(), **kwargs)


def loader_kwargs(num_workers: int) -> dict:
    import torch
    return {"num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": num_workers > 0}
