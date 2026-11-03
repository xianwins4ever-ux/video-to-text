"""下载 faster-whisper 模型到 D 盘（走 hf-mirror.com，因为 huggingface.co 被墙）。

用法：
    python fetch_models.py large-v3
    python fetch_models.py large-v3 medium small
"""
import os
import sys
import time

MODELS_ROOT = r"D:\tools\v2t\models"

# 必须在 import faster_whisper 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HOME"] = MODELS_ROOT
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# 关键：HF 新的 Xet 存储后端会把大文件重定向到 cas-server.xethub.hf.co，
# 该域名未被 hf-mirror 镜像，会 401。禁用 Xet 强制走传统 resolve 下载。
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

from faster_whisper import download_model  # noqa: E402


def fetch(name: str) -> str:
    print(f"[fetch] {name} -> {MODELS_ROOT} (endpoint={os.environ['HF_ENDPOINT']})", flush=True)
    t0 = time.time()
    path = download_model(name, cache_dir=MODELS_ROOT)
    dt = time.time() - t0
    print(f"[done ] {name} in {dt:.1f}s -> {path}", flush=True)
    return path


if __name__ == "__main__":
    names = sys.argv[1:] or ["large-v3"]
    ok, bad = [], []
    for n in names:
        try:
            ok.append(fetch(n))
        except Exception as e:  # noqa: BLE001
            bad.append((n, f"{type(e).__name__}: {e}"))
            print(f"[FAIL ] {n}: {type(e).__name__}: {e}", flush=True)
    print("\n===== SUMMARY =====")
    for p in ok:
        print("OK  ", p)
    for n, e in bad:
        print("FAIL", n, e)
    sys.exit(1 if bad else 0)
