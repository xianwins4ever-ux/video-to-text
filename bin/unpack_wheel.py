"""解包 CUDA wheel 到 pylib\\nvidia\\<lib>\\bin\\（模拟 pip install --target 的布局）。"""
import os
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

WHEEL_DIR = str(paths.WHEELS)
PYLIB = str(paths.PYLIB)


def extract(wheel: str) -> list[str]:
    out = []
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            # wheel 内路径形如 nvidia/cublas/bin/cublas64_12.dll
            if name.startswith("nvidia/") and name.endswith(".dll"):
                dest = os.path.join(PYLIB, name.replace("/", os.sep))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with z.open(name) as src, open(dest, "wb") as f:
                    f.write(src.read())
                out.append(f"{name}  ->  {os.path.getsize(dest)/1e6:.1f} MB")
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or [f for f in os.listdir(WHEEL_DIR) if f.endswith(".whl")]
    for n in names:
        p = os.path.join(WHEEL_DIR, n) if not os.path.isabs(n) else n
        if not os.path.exists(p):
            print(f"[skip] {p} 不存在")
            continue
        print(f"==== {os.path.basename(p)} ====")
        for line in extract(p):
            print("  ", line)
