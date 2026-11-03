# -*- coding: utf-8 -*-
"""统一的路径与环境探测 —— 全项目唯一的路径真相源。

为什么要这个模块
----------------
早期各脚本各写各的 `Path(r"D:\\tools\\v2t")`，作者机器上能跑，
别人 clone 下来必挂。现在所有脚本都从这里导入，改一处即全局生效。

解析优先级
----------
1. 环境变量（V2T_HOME / V2T_FFMPEG_DIR / V2T_MODELS_DIR / V2T_COOKIE_DIR）
2. 按本文件位置推导（bin/paths.py -> 上一级即项目根）
3. 常见安装位置兜底

依赖外部程序时（ffmpeg）额外查 PATH，找不到不报错、返回 None，
由调用方决定降级策略。
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------- 项目根

def _detect_home() -> Path:
    env = os.environ.get("V2T_HOME")
    if env:
        return Path(env).expanduser().resolve()
    # bin/paths.py -> bin -> 项目根
    return Path(__file__).resolve().parent.parent


HOME: Path = _detect_home()
BIN: Path = HOME / "bin"
PYLIB: Path = HOME / "pylib"
MODELS: Path = Path(os.environ.get("V2T_MODELS_DIR") or (HOME / "models")).expanduser()
WHEELS: Path = HOME / "wheels"
DATA: Path = HOME / "data"
ASSETS: Path = HOME / "assets"
VENV_QWEN: Path = HOME / "venv_qwen"

# 把 pylib 挂进 sys.path（yt-dlp / opencc / CUDA dll 都在这里，不污染全局环境）
if PYLIB.is_dir() and str(PYLIB) not in sys.path:
    sys.path.insert(0, str(PYLIB))


# ---------------------------------------------------------------- ffmpeg

def _detect_ffmpeg_dir() -> Path | None:
    """返回含 ffmpeg 可执行文件的目录；找不到返回 None。"""
    env = os.environ.get("V2T_FFMPEG_DIR") or os.environ.get("FFMPEG_DIR")
    if env:
        p = Path(env).expanduser()
        if (p / "ffmpeg.exe").exists() or (p / "ffmpeg").exists():
            return p.resolve()

    exe = shutil.which("ffmpeg")
    if exe:
        return Path(exe).resolve().parent

    for cand in (
        HOME / "ffmpeg",
        HOME / "tools" / "ffmpeg",
        Path(r"D:\hyperframes-tools\ffmpeg"),   # 作者机器上的旧位置，保留兼容
        Path(r"C:\ffmpeg\bin"),
        Path("/usr/bin"),
        Path("/usr/local/bin"),
        Path("/opt/homebrew/bin"),
    ):
        if (cand / "ffmpeg.exe").exists() or (cand / "ffmpeg").exists():
            return cand
    return None


FFMPEG_DIR: Path | None = _detect_ffmpeg_dir()


def ffmpeg_exe() -> str:
    """ffmpeg 可执行文件路径。找不到时返回 "ffmpeg"（交给 PATH 解析）。"""
    if FFMPEG_DIR:
        p = FFMPEG_DIR / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if p.exists():
            return str(p)
    return "ffmpeg"


def ffprobe_exe() -> str:
    if FFMPEG_DIR:
        p = FFMPEG_DIR / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if p.exists():
            return str(p)
    return "ffprobe"


def require_ffmpeg() -> str:
    """需要 ffmpeg 但可能缺失的场合调用，给出可执行的修复建议。"""
    exe = ffmpeg_exe()
    if exe == "ffmpeg" and not shutil.which("ffmpeg"):
        raise RuntimeError(
            "找不到 ffmpeg。请任选一种方式：\n"
            "  1) 装到系统 PATH（Windows: winget install Gyan.FFmpeg）\n"
            "  2) 解压到 <v2t目录>/ffmpeg/\n"
            "  3) 设置环境变量 V2T_FFMPEG_DIR 指向 ffmpeg 所在目录"
        )
    return exe


# ---------------------------------------------------------------- cookie

def cookie_dir() -> Path:
    """登录凭证目录（只在本机，不入库）。"""
    env = os.environ.get("V2T_COOKIE_DIR")
    d = Path(env).expanduser() if env else Path.home() / ".v2t" / "cookies"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- 模型

def model_dir(name: str) -> Path | None:
    """按名字找本地模型目录，支持 faster-whisper 的几种缓存命名。"""
    for cand in (
        MODELS / name,
        MODELS / f"faster-whisper-{name}",
        MODELS / f"models--Systran--faster-whisper-{name}",
    ):
        if cand.is_dir():
            return cand
    return None


# ---------------------------------------------------------------- 自检

def describe() -> str:
    """供 doctor / 排错时打印。"""
    lines = [
        f"V2T_HOME     = {HOME}",
        f"PYLIB        = {PYLIB}  ({'存在' if PYLIB.is_dir() else '缺失'})",
        f"MODELS       = {MODELS}  ({'存在' if MODELS.is_dir() else '缺失'})",
        f"FFMPEG_DIR   = {FFMPEG_DIR or '未找到'}",
        f"COOKIE_DIR   = {cookie_dir()}",
        f"VENV_QWEN    = {VENV_QWEN}  ({'存在' if VENV_QWEN.is_dir() else '缺失'})",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
