"""下载 CUDA 运行库 wheel（ctranslate2 的 CUDA 推理需要 cublas64_12.dll + cudnn64_9.dll）。

curl 的 -C - 在目标文件不存在/0 字节时会卡住不写入，这里用 urllib 自己实现
带断点续传和进度显示的下载。
"""
import os
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

WHEEL_DIR = str(paths.WHEELS)

WHEELS = {
    "cublas": "https://mirrors.aliyun.com/pypi/packages/20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
    "cudnn": "https://mirrors.aliyun.com/pypi/packages/6f/5c/f77147ce7e27a4e9087fb34b0539ff085c68e7093e96ee85576fe31fe064/nvidia_cudnn_cu12-9.9.0.52-py3-none-win_amd64.whl",
}


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def fetch(name: str, url: str, tries: int = 12) -> str:
    os.makedirs(WHEEL_DIR, exist_ok=True)
    out = os.path.join(WHEEL_DIR, url.rsplit("/", 1)[-1])

    for attempt in range(1, tries + 1):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            **({"Range": f"bytes={have}-"} if have else {}),
        })
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length", 0)) + have
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                t0, done = time.time(), have
                with open(out, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        el = time.time() - t0
                        sp = (done - have) / el if el > 0 else 0
                        pct = done / total * 100 if total else 0
                        eta = (total - done) / sp if sp > 0 and total else 0
                        print(f"\r[{name}] {pct:5.1f}%  {human(done)}/{human(total)}  "
                              f"{human(sp)}/s  ETA {eta/60:.1f}min   ", end="", flush=True)
                print()
            if total and os.path.getsize(out) >= total:
                print(f"[done] {name}: {human(os.path.getsize(out))}")
                return out
            print(f"[warn] {name}: 未下完（{human(os.path.getsize(out))}/{human(total)}），续传…")
        except Exception as e:  # noqa: BLE001
            print(f"\n[retry] {name} 第 {attempt} 次出错：{type(e).__name__}: {e}")
            time.sleep(min(30, 3 * attempt))

    raise RuntimeError(f"{name} 下载失败")


if __name__ == "__main__":
    want = sys.argv[1:] or list(WHEELS)
    for k in want:
        fetch(k, WHEELS[k])
    print("\n===== 结果 =====")
    for f in os.listdir(WHEEL_DIR):
        print(f"  {f}  {human(os.path.getsize(os.path.join(WHEEL_DIR, f)))}")
