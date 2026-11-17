"""
kaggle_push.py
==============
Đẩy code + data lên Kaggle, cho notebooks/main.ipynb tự chạy trên GPU, in link theo dõi.

    python -m vnlp.kaggle_push            # chạy thật (quick_test=false, bật detection + pipeline)
    python -m vnlp.kaggle_push quick      # chạy thử vài ảnh
    python -m vnlp.kaggle_push pull       # tải kết quả lần chạy mới nhất về outputs/<experiment>/

Lần đầu: xem README mục "Bắt đầu nhanh" (kaggle auth login, xác minh số điện thoại).

Lên Kaggle (đều PRIVATE):
  data   : config kaggle.data_dataset, hoặc <user>/cs406-vnlp-data do script tự upload
           (chỉ upload lại khi manifest CSV đổi)
  code   : <user>/cs406-vnlp-code = vnlp/ + config.yaml + BUILD_INFO.json (upload mỗi lần,
           ghi chú version = tên experiment + git commit)
  kernel : <user>/<kaggle.kernel_slug>-<experiment>: mỗi experiment 1 notebook riêng (chạy song song được),
           mỗi lần push là 1 version mới (tab Versions trên Kaggle)
"""

import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from vnlp import config

DATA_SLUG, CODE_SLUG = "cs406-vnlp-data", "cs406-vnlp-code"
STAGING = config.OUTPUT_ROOT / "_kaggle_upload"
STATE_FILE = STAGING / "state.json"


# ── Kaggle API ────────────────────────────────────────────────────────
def get_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("Chưa cài Kaggle CLI: pip install kaggle")
    api = KaggleApi()
    try:
        api.authenticate()
    except (Exception, SystemExit):
        pass
    user = api.get_config_value("username")
    if not user:
        sys.exit("Chưa đăng nhập Kaggle. Chạy: kaggle auth login   (xem README mục 'Bắt đầu nhanh')")
    return api, user


def dataset_exists(api, ref):
    try:
        api.dataset_status(ref)
        return True
    except Exception:
        return False


def write_meta(folder: Path, user, slug, title):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")


def upload_dataset(api, folder, ref, note):
    """Tạo mới (private) hoặc thêm version. dir_mode=zip: thư mục con được nén khi upload."""
    if dataset_exists(api, ref):
        api.dataset_create_version(str(folder), version_notes=note, quiet=True, dir_mode="zip")
    else:
        api.dataset_create_new(str(folder), public=False, quiet=True, dir_mode="zip")


def wait_ready(api, ref, timeout=1200):
    start = time.time()
    while time.time() - start < timeout:
        try:
            status = str(api.dataset_status(ref)).lower()
        except Exception:
            status = "pending"
        if "ready" in status:
            return
        if "error" in status:
            sys.exit(f"Kaggle báo lỗi khi xử lý dataset {ref}")
        time.sleep(10)
    sys.exit(f"Quá {timeout}s mà dataset {ref} chưa sẵn sàng – xem trên kaggle.com rồi chạy lại")


# ── Data ──────────────────────────────────────────────────────────────
def push_data(api, user, state):
    shared = config.CFG.get("kaggle", {}).get("data_dataset")
    if shared:
        if not dataset_exists(api, shared):
            sys.exit(f"Không truy cập được dataset {shared}. Nhờ chủ dataset share (Settings > Collaborators).")
        print(f"[data] dùng dataset được share: {shared}")
        return shared

    if not (config.RECOG_MANIFEST.is_file() and config.DET_MANIFEST.is_file() and config.DET_DATA_YAML.is_file()):
        sys.exit(f"Chưa có split '{config.SPLIT_NAME}' trên máy. Chạy 2 lệnh split (README), "
                 "hoặc đặt kaggle.data_dataset trong config.yaml = dataset được share.")
    # Upload MỌI bộ split đang có (splits/, splits_lodo-*/) -> mỗi experiment chọn bộ của mình
    manifests = sorted(config.DATA_DIR.glob("*_split*.csv"))
    split_dirs = sorted(p for p in config.DATA_DIR.glob("splits*") if p.is_dir())
    fp = hashlib.md5(b"".join(p.name.encode() + p.read_bytes() for p in manifests)).hexdigest()
    ref = f"{user}/{DATA_SLUG}"
    if state.get("data_fingerprint") == fp and dataset_exists(api, ref):
        print(f"[data] không đổi -> dùng lại {ref}")
        return ref

    folder = STAGING / "data"
    shutil.rmtree(folder, ignore_errors=True)
    write_meta(folder, user, DATA_SLUG, "CS406 VNLP Data")
    for d in split_dirs:
        shutil.copytree(d, folder / d.name)
    for p in manifests:
        shutil.copy2(p, folder / p.name)
    print(f"[data] đang upload {ref} ({', '.join(d.name for d in split_dirs)}; ~250 MB/bộ, sẽ lâu)...")
    upload_dataset(api, folder, ref, f"data {fp[:8]}")
    state["data_fingerprint"] = fp
    STATE_FILE.write_text(json.dumps(state, indent=1))  # lưu ngay: lỗi bước sau không phải upload lại
    return ref


# ── Code ──────────────────────────────────────────────────────────────
def build_info():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=config.REPO_ROOT, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    commit = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    return {"experiment": config.EXPERIMENT,
            "git_commit": f"{commit}{'+dirty' if dirty else ''}" if commit else None,
            "git_branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
            "pushed_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def push_code(api, user, info):
    ref = f"{user}/{CODE_SLUG}"
    folder = STAGING / "code"
    shutil.rmtree(folder, ignore_errors=True)
    write_meta(folder, user, CODE_SLUG, "CS406 VNLP Code")
    root = config.REPO_ROOT
    shutil.copytree(root / "vnlp", folder / "vnlp", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(root / "requirements-kaggle.txt", folder / "requirements-kaggle.txt")
    # Upload config THỰC TẾ đang dùng (config.yaml + mọi config.reload(...) ghi đè),
    # để thí nghiệm chạy được mà không phải sửa tạm config.yaml.
    with open(folder / "config.yaml", "w", encoding="utf-8") as f:
        print("# Sinh bởi vnlp.kaggle_push từ config đang dùng (đã gồm ghi đè)", file=f)
        yaml.safe_dump(config.CFG, f, allow_unicode=True, sort_keys=False)
    (folder / "BUILD_INFO.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")

    note = f"{info['experiment']} | {info['git_commit'] or 'no-git'} | {info['pushed_at']}"
    print(f"[code] đang upload {ref} ({note})...")
    upload_dataset(api, folder, ref, note)
    return ref


# ── Kernel ────────────────────────────────────────────────────────────
def kernel_ref(user):
    """<user>/<kernel_slug>-<experiment> (slug Kaggle: chữ thường, số, gạch ngang)."""
    base = config.CFG.get("kaggle", {}).get("kernel_slug", "cs406-vnlp")
    exp = re.sub(r"[^a-z0-9]+", "-", config.EXPERIMENT.lower()).strip("-") or "default"
    return f"{user}/{base}-{exp}"


def push_kernel(api, user, sources, quick):
    folder = STAGING / "kernel"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)

    overrides = {"notebook.quick_test": quick, "notebook.run_detection": True, "notebook.run_pipeline": True}
    nb = json.loads((config.REPO_ROOT / "notebooks" / "main.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and "OVERRIDES = {" in src:
            inject = "".join(f"    {k!r}: {v!r},\n" for k, v in overrides.items())
            cell["source"] = src.replace("OVERRIDES = {\n", "OVERRIDES = {\n" + inject, 1).splitlines(keepends=True)
            break
    (folder / "main.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    ref = kernel_ref(user)
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": ref, "title": ref.split("/")[1].replace("-", " ").upper(), "code_file": "main.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": sources, "competition_sources": [], "kernel_sources": [],
    }, indent=1), encoding="utf-8")

    print(f"[kernel] đang push {ref} ({'quick' if quick else 'chạy thật'})...")
    resp = api.kernels_push(str(folder))
    if getattr(resp, "error", None):
        sys.exit(f"Kaggle từ chối notebook: {resp.error}")
    return getattr(resp, "url", None) or f"https://www.kaggle.com/code/{ref}"


def push(quick):
    api, user = get_api()
    STAGING.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.is_file() else {}
    info = build_info()
    if info["git_commit"] and info["git_commit"].endswith("+dirty"):
        print("⚠️  Có thay đổi chưa commit – vẫn chạy, nhưng nên commit để sau này truy lại được code.")

    data_ref = push_data(api, user, state)
    code_ref = push_code(api, user, info)
    print("[wait] đợi Kaggle xử lý dataset...")
    wait_ready(api, data_ref)
    wait_ready(api, code_ref)
    url = push_kernel(api, user, [data_ref, code_ref], quick)

    print(f"\n✅ Đã gửi experiment '{info['experiment']}' (commit {info['git_commit']}). Theo dõi tại:")
    print(f"   {url}")
    print("   Logs: tiến trình | Output: kết quả | Versions: các lần chạy trước")
    print("   Chạy xong tải kết quả về: python -m vnlp.kaggle_push pull")


def pull():
    """Tải outputs/ của version mới nhất về máy (giữ nguyên outputs/<experiment>/...)."""
    api, user = get_api()
    ref = kernel_ref(user)
    resp = api.kernels_status(ref)
    status = str(getattr(resp, "status", resp))
    print(f"[pull] {ref}: {status}")
    if "complete" not in status.lower():
        sys.exit("Notebook chưa chạy xong (hoặc lỗi) – xem trên Kaggle rồi pull lại sau.")

    tmp = STAGING / "pull"
    shutil.rmtree(tmp, ignore_errors=True)
    api.kernels_output(ref, str(tmp), file_pattern=r"^outputs/", force=True)
    src = tmp / "outputs"
    if not src.is_dir():
        sys.exit("Không thấy thư mục outputs/ trong kết quả Kaggle.")
    for exp_dir in src.iterdir():
        dst = config.OUTPUT_ROOT / exp_dir.name
        shutil.copytree(exp_dir, dst, dirs_exist_ok=True)
        print(f"[pull] -> {dst}")
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "pull":
        pull()
    elif cmd in ("run", "quick"):
        push(quick=cmd == "quick")
    else:
        sys.exit("Dùng: python -m vnlp.kaggle_push [quick|pull]")


if __name__ == "__main__":
    main()
