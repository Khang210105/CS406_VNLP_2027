"""
kaggle_push.py
==============
Đẩy code + data lên Kaggle, cho notebooks/main.ipynb tự chạy trên GPU, in link theo dõi.

    python -m vnlp.kaggle_push            # chạy thật (quick_test=false; phần nào chạy do config notebook.run_*)
    python -m vnlp.kaggle_push quick      # chạy thử vài ảnh
    python -m vnlp.kaggle_push pull       # tải kết quả lần chạy mới nhất về outputs/<experiment>/

Lần đầu: xem README mục "Bắt đầu nhanh" (kaggle auth login, xác minh số điện thoại).

Lên Kaggle (đều PRIVATE):
  data   : <user>/cs406-data-<bộ split>: MỖI bộ split (splits/, splits_lodo-.../) 1 dataset riêng,
           chỉ upload bộ experiment đang dùng và chỉ khi manifest đổi.
           kaggle.data_owner = người khác -> dùng dataset họ đã share, không cần data trên máy.
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

CODE_SLUG = "cs406-vnlp-code"
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


def dataset_version(api, ref):
    """(status, current_version_number); (None, 0) nếu chưa tồn tại."""
    try:
        info = json.loads(api.dataset_status(ref, format="json"))
        return str(info.get("status", "")).lower(), int(info.get("current_version_number") or 0)
    except Exception:
        return None, 0


def upload_dataset(api, folder, ref, note):
    """Tạo mới (private) hoặc thêm version. dir_mode=zip: thư mục con được nén khi upload.
    Trả về số version cần đợi (để notebook không gắn nhầm version cũ)."""
    _, before = dataset_version(api, ref)
    if before:
        api.dataset_create_version(str(folder), version_notes=note, quiet=True, dir_mode="zip")
    else:
        api.dataset_create_new(str(folder), public=False, quiet=True, dir_mode="zip")
    return before + 1


def wait_ready(api, ref, min_version=0, timeout=1800):
    """Đợi tới khi dataset có version >= min_version VÀ ready.
    (Chỉ xem 'ready' là chưa đủ: trong lúc version mới đang xử lý, Kaggle vẫn báo ready của version cũ
    -> notebook gắn nhầm data/code cũ.)"""
    start = time.time()
    while time.time() - start < timeout:
        status, version = dataset_version(api, ref)
        if status and "error" in status:
            sys.exit(f"Kaggle báo lỗi khi xử lý dataset {ref}")
        if status and "ready" in status and version >= min_version:
            return version
        time.sleep(10)
    sys.exit(f"Quá {timeout}s mà dataset {ref} chưa lên version {min_version} – xem trên kaggle.com rồi chạy lại")


# ── Data ──────────────────────────────────────────────────────────────
def data_slug():
    """Mỗi bộ split 1 dataset riêng -> chỉ upload bộ experiment đang dùng, bộ khác không bị upload lại.
    splits -> cs406-data-std | splits_lodo-view1-view2_tr-view3x100-view4x100 -> cs406-data-lodo-v1-v2-tr-v3x100-v4x100"""
    name = config.SPLIT_NAME + config.DET_SPLIT_TAG
    short = name.removeprefix("splits").replace("view", "v").replace("_", "-").strip("-").lower() or "std"
    short = re.sub(r"[^a-z0-9-]+", "-", short)
    slug = f"cs406-data-{short}"
    if len(slug) > 50:  # giới hạn slug Kaggle
        slug = f"cs406-data-{hashlib.md5(name.encode()).hexdigest()[:10]}"
    return slug, name


def push_data(api, user, state):
    slug, name = data_slug()
    owner = config.CFG.get("kaggle", {}).get("data_owner")
    if owner and owner != user:
        ref = f"{owner}/{slug}"
        if not dataset_exists(api, ref):
            sys.exit(f"Không truy cập được {ref} (bộ split '{name}'). Nhờ {owner} chạy push 1 lần với "
                     f"config này rồi share dataset cho bạn (Settings > Collaborators).")
        print(f"[data] dùng dataset được share: {ref}")
        return ref, 0

    det_dir = config.DET_SPLIT_DIR
    needed = [config.RECOG_MANIFEST, config.DET_MANIFEST, config.RECOG_SPLIT_DIR, config.DET_DATA_YAML]
    missing = [str(p.relative_to(config.DATA_DIR)) for p in needed if not p.exists()]
    if missing:
        sys.exit(f"Chưa có bộ split '{name}' trên máy (thiếu {missing}). Chạy 2 lệnh split (README), "
                 "hoặc đặt kaggle.data_owner trong config.yaml để dùng data được share.")

    manifests = [config.RECOG_MANIFEST, config.DET_MANIFEST]
    fp = hashlib.md5(b"".join(p.name.encode() + p.read_bytes() for p in manifests)).hexdigest()
    ref = f"{user}/{slug}"
    fingerprints = state.setdefault("data_fingerprints", {})
    if fingerprints.get(ref) == fp and dataset_exists(api, ref):
        print(f"[data] bộ '{name}' không đổi -> dùng lại {ref}")
        return ref, 0

    # Giữ nguyên đường dẫn tương đối như dataset/ trên máy -> notebook + config tìm đúng
    folder = STAGING / "data"
    shutil.rmtree(folder, ignore_errors=True)
    write_meta(folder, user, slug, f"CS406 data {name}"[:50])
    shutil.copytree(config.RECOG_SPLIT_DIR, folder / config.RECOG_SPLIT_DIR.relative_to(config.DATA_DIR))
    shutil.copytree(det_dir, folder / det_dir.relative_to(config.DATA_DIR))
    for p in manifests:
        shutil.copy2(p, folder / p.name)
    print(f"[data] đang upload {ref} (bộ '{name}')...")
    need = upload_dataset(api, folder, ref, f"{name} {fp[:8]}")
    fingerprints[ref] = fp
    STATE_FILE.write_text(json.dumps(state, indent=1))  # lưu ngay: lỗi bước sau không phải upload lại
    return ref, need


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
    return ref, upload_dataset(api, folder, ref, note)


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

    # Chỉ ép quick_test; chạy phần nào (crnn/parseq/detection/pipeline) do config notebook.run_* quyết định
    overrides = {"notebook.quick_test": quick}
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

    data_ref, data_ver = push_data(api, user, state)
    code_ref, code_ver = push_code(api, user, info)
    print("[wait] đợi Kaggle xử lý xong version mới của dataset...")
    print(f"       {data_ref} v{wait_ready(api, data_ref, data_ver)} | {code_ref} v{wait_ready(api, code_ref, code_ver)}")
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
