#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v2t —— 视频链接/本地文件 → 文案（带时间戳 md + 纯文本 txt + srt 字幕）

策略：优先抓平台自带字幕（秒级、零识别误差），没有字幕才跑本地 ASR。

本地 ASR 链路：ffmpeg → 16k mono wav → numpy → faster-whisper(CUDA float16)
注意：故意不走 faster-whisper 自带的音频解码（PyAV 19.0.1 与 faster-whisper 1.2.1
不兼容，会抛 TypeError: open() got an unexpected keyword argument 'metadata_errors'）。

用法：
    python v2t.py <url 或 本地文件> [选项]

常用：
    python v2t.py "https://www.bilibili.com/video/BVxxxx"
    python v2t.py "https://youtu.be/xxxx" --model large-v3
    python v2t.py D:\\video.mp4 --outdir D:\\out
    python v2t.py <url> --cookies-from-browser edge     # 抓 B站 AI 字幕需要登录
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

# ---------------------------------------------------------------- 环境
# 路径自适应：优先环境变量，其次按脚本位置推导，最后才用旧默认值。
# 这样别人 clone 下来解压到任意目录都能跑，不依赖作者机器的路径。
def _detect_tools() -> Path:
    env = os.environ.get("V2T_HOME")
    if env:
        return Path(env).expanduser().resolve()
    # bin/v2t.py -> 上一级即工具根目录
    return Path(__file__).resolve().parent.parent


def _detect_ffmpeg_dir() -> Path:
    """找 ffmpeg 所在目录：环境变量 > PATH > 工具目录内 > 常见安装位置。"""
    env = os.environ.get("V2T_FFMPEG_DIR") or os.environ.get("FFMPEG_DIR")
    if env and (Path(env) / "ffmpeg.exe").exists():
        return Path(env).resolve()
    if env and (Path(env) / "ffmpeg").exists():
        return Path(env).resolve()

    exe = shutil.which("ffmpeg")
    if exe:
        return Path(exe).resolve().parent

    for cand in (
        Path(__file__).resolve().parent.parent / "ffmpeg",
        Path(r"D:\hyperframes-tools\ffmpeg"),
        Path(r"C:\ffmpeg\bin"),
        Path("/usr/bin"),
        Path("/usr/local/bin"),
        Path("/opt/homebrew/bin"),
    ):
        if (cand / "ffmpeg.exe").exists() or (cand / "ffmpeg").exists():
            return cand
    return Path(".")


TOOLS = _detect_tools()
PYLIB = TOOLS / "pylib"
MODELS = TOOLS / "models"
FFMPEG_DIR = _detect_ffmpeg_dir()
FFMPEG = FFMPEG_DIR / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
FFPROBE = FFMPEG_DIR / ("ffprobe.exe" if os.name == "nt" else "ffprobe")

if PYLIB.is_dir():
    sys.path.insert(0, str(PYLIB))

# 浏览器嗅探通道（抖音/快手/小红书）。可选依赖：playwright 缺失时自动降级为不可用，
# 不影响其他站点的正常流程。
try:
    sys.path.insert(0, str(TOOLS / "bin"))
    import browser_fetch  # noqa: E402
except Exception as _e:  # noqa: BLE001
    browser_fetch = None

# B站原生字幕直取（需登录）。yt-dlp 不带登录态会误判"无字幕"，这条路径补上。
try:
    sys.path.insert(0, str(TOOLS / "bin"))
    import bili_subs  # noqa: E402
except Exception as _e:  # noqa: BLE001
    bili_subs = None
    _BROWSER_IMPORT_ERR = _e

# ctranslate2 的 CUDA 推理需要 cublas64_12.dll / cudnn64_9.dll。
# nvidia-* wheel 把它们放在 pylib/nvidia/<lib>/bin/，不在 PATH 里，
# 必须用 os.add_dll_directory 注册，且要在 import ctranslate2 之前完成。
#
# 坑：add_dll_directory 返回的句柄一旦被 GC，目录注册就失效。
# 所以必须把句柄存进模块级列表里一直持有，否则表现为
# "cublas64_12.dll is not found or cannot be loaded"（文件明明在）。
#
# 坑二（2026-11-01 实测踩到）：只做 add_dll_directory 还不够。
# ctranslate2 的 C++ 加载器用的是系统 LoadLibrary 搜索顺序，不走 Python 的
# 扩展模块目录表，于是 cublas 链在真实转写时仍然报
# "Library cublas64_12.dll is not found or cannot be loaded"。
# 必须同时把这些目录塞进 PATH——这才是 cublas 能被找到的决定性一步。
# （当年"CUDA 测试通过"是假阳性：只加载了模型、转写的是静音，没走到 encode。）
_DLL_DIR_HANDLES = []
if os.name == "nt":
    _dll_dirs = []
    for _sub in ("cublas", "cudnn", "cuda_runtime", "cudart", "cufft", "curand",
                 "cusolver", "cusparse", "nvjitlink", "nvrtc"):
        _d = PYLIB / "nvidia" / _sub / "bin"
        if _d.is_dir():
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(_d)))
            except Exception:
                pass
            _dll_dirs.append(str(_d))
    if _dll_dirs:
        os.environ["PATH"] = ";".join(_dll_dirs) + ";" + os.environ.get("PATH", "")

# 模型缓存必须落 D 盘（C 盘只剩几个 GB），且 huggingface.co 被墙 → 走镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HOME"] = str(MODELS)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# 进度回调：CLI 下没人监听，只 print 到 stderr；WebUI 下由 webui.py 注入，
# 每条日志连同进度百分比一起推给浏览器。
_EMIT = None

# 实际使用的引擎（"qwen" / "whisper"），供结果元信息展示
_ENGINE_USED = "whisper"


def set_emitter(fn) -> None:
    """供 WebUI 注入回调：每产生一条日志就调用 fn(dict)。传 None 取消。"""
    global _EMIT
    _EMIT = fn


def emit(event: dict) -> None:
    if _EMIT is None:
        return
    try:
        _EMIT(event)
    except Exception:
        pass  # 回调方的问题不该拖垮转写


def log(msg: str, pct: float | None = None) -> None:
    print(msg, file=sys.stderr, flush=True)
    if pct is not None:
        emit({"type": "progress", "msg": msg, "pct": pct})
    else:
        emit({"type": "log", "msg": msg})


def stage(name: str, pct: float) -> None:
    """推进阶段标签（WebUI 顶部进度条用）。"""
    emit({"type": "stage", "stage": name, "pct": pct})


# ---------------------------------------------------------------- 字幕
def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return text.strip()


def parse_json3(path: Path) -> list[dict]:
    """YouTube json3 字幕格式。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict] = []
    for ev in data.get("events", []):
        segs = ev.get("segs") or []
        text = _clean("".join(s.get("utf8", "") for s in segs))
        if not text:
            continue
        start = float(ev.get("tStartMs", 0)) / 1000.0
        end = start + float(ev.get("dDurationMs", 0)) / 1000.0
        out.append({"start": start, "end": end, "text": text})
    return out


def parse_bilibili_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for item in data.get("body", []):
        content = _clean(str(item.get("content", "")))
        if content:
            out.append({"start": float(item.get("from", 0.0)), "end": float(item.get("to", 0.0)), "text": content})
    return out


def parse_vtt(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    out: list[dict] = []
    for block in re.split(r"\n\s*\n", raw):
        lines = [l for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        ts_idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if ts_idx is None:
            continue

        def sec(v: str) -> float:
            v = v.strip().replace(",", ".")
            parts = v.split(":")
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                return 0.0
            while len(nums) < 3:
                nums.insert(0, 0.0)
            return nums[0] * 3600 + nums[1] * 60 + nums[2]

        a, _, b = lines[ts_idx].partition("-->")
        text = _clean(" ".join(lines[ts_idx + 1:]))
        if text:
            out.append({"start": sec(a), "end": sec(b.split()[0] if b.split() else "0"), "text": text})
    return out


def parse_srt(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    out: list[dict] = []
    for block in re.split(r"\n\s*\n", raw):
        lines = [l for l in block.splitlines() if l.strip()]
        ts_idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if ts_idx is None:
            continue

        def sec(v: str) -> float:
            v = v.strip().replace(",", ".")
            parts = v.split(":")
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                return 0.0
            while len(nums) < 3:
                nums.insert(0, 0.0)
            return nums[0] * 3600 + nums[1] * 60 + nums[2]

        a, _, b = lines[ts_idx].partition("-->")
        text = _clean(" ".join(lines[ts_idx + 1:]))
        if text:
            out.append({"start": sec(a), "end": sec(b.split()[0] if b.split() else "0"), "text": text})
    return out


def load_cues(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        # 先试 json3（YouTube），再试 B站 格式
        try:
            cues = parse_json3(path)
            if cues:
                return cues
        except Exception:
            pass
        try:
            return parse_bilibili_json(path)
        except Exception:
            return []
    if suffix == ".json3":
        try:
            return parse_json3(path)
        except Exception:
            return []
    if suffix == ".vtt":
        return parse_vtt(path)
    if suffix in (".srt", ".ass"):
        return parse_srt(path)
    try:
        return parse_vtt(path)
    except Exception:
        return []


# ---------------------------------------------------------------- 下载
def ytdlp_opts(args, extra: dict | None = None) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ffmpeg_location": str(FFMPEG_DIR),
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
        "noplaylist": not args.playlist,
    }
    # 浏览器 cookie 数据库在浏览器运行时是锁着的（yt-dlp 已知问题）。
    # 这里先探测可读性，读不到就退回匿名模式，并给出可操作的提示，
    # 而不是让整个流程崩在栈里。
    if args.cookies_from_browser and not getattr(args, "_cookies_broken", False):
        try:
            import yt_dlp
            from yt_dlp.cookies import extract_cookies_from_browser
            extract_cookies_from_browser(args.cookies_from_browser)
            opts["cookiesfrombrowser"] = (args.cookies_from_browser,)
        except Exception as e:  # noqa: BLE001
            args._cookies_broken = True
            log(f"[cookies] 无法读取 {args.cookies_from_browser} 的 cookies（{type(e).__name__}）。")
            log(f"[cookies] 常见原因：浏览器正在运行，cookie 数据库被锁。")
            log(f"[cookies] 处理：① 完全退出 {args.cookies_from_browser} 后重试；"
                f"或 ② 导出 cookies.txt 后用 --cookies <文件>；"
                f"或 ③ 先不管（匿名模式，B站 AI 字幕可能拿不到）")
    if args.cookies:
        opts["cookiefile"] = args.cookies
    if args.proxy:
        opts["proxy"] = args.proxy
    if extra:
        opts.update(extra)
    return opts


def retry(fn, tries: int = 4, delay: float = 2.0, label: str = ""):
    """网络抖动重试。YouTube 等站点在国内会随机 SSL 断流。"""
    last = None
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < tries:
                log(f"[retry] {label} 第 {i} 次失败（{type(e).__name__}），{delay:.0f}s 后重试…")
                time.sleep(delay * i)
    raise last  # type: ignore[misc]


def probe_and_fetch_subs(url: str, args, workdir: Path) -> tuple[dict, list[dict]]:
    """返回 (info, cues)。cues 非空表示拿到了平台字幕。

    两步走：先只取元数据看有哪些字幕语言（避免为不存在的语言触发 429 白重试），
    再只下载真正存在的那一种。
    """
    import yt_dlp

    subdir = workdir / "subs"
    subdir.mkdir(parents=True, exist_ok=True)

    # ---- 第一步：只取元数据
    meta_opts = ytdlp_opts(args, {"skip_download": True})

    def run_meta():
        with yt_dlp.YoutubeDL(meta_opts) as ydl:
            return ydl.extract_info(url, download=False)

    info = retry(run_meta, tries=args.retries, label="元数据探测")
    if info and info.get("entries"):
        info = info["entries"][0]

    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    available = list(manual.keys()) + [k for k in auto.keys() if k not in manual]

    if not available:
        log("[subs] 该视频没有任何字幕轨 → 转本地 ASR")
        return info, []

    # ---- 选语言：优先用户指定，其次中文，最后英文/第一个
    def pick(cands: list[str]) -> str | None:
        for want in (args.sub_lang, "zh-Hans", "zh-CN", "zh", "zh-Hant", "en", "en-US"):
            for a in cands:
                if a == want or a.startswith(want.split("-")[0] + "-") or a == want.split("-")[0]:
                    return a
        return cands[0] if cands else None

    # 人工字幕优先于自动字幕（人工更准）
    chosen = pick(list(manual.keys())) if manual else None
    is_auto = False
    if not chosen:
        chosen = pick(available)
        is_auto = True
    if not chosen:
        return info, []

    log(f"[subs] 可选 {len(available)} 种语言，选用 {chosen}"
        f"（{'自动生成' if is_auto else '人工上传'}）")

    # ---- 第二步：只下这一种
    sub_opts = ytdlp_opts(args, {
        "skip_download": True,
        "writesubtitles": not is_auto,
        "writeautomaticsub": is_auto,
        "subtitleslangs": [chosen],
        # json3 优先：实测 YouTube 的 vtt/srt 端点在国内经常 SSL 断流，json3 更稳
        "subtitlesformat": "json3/vtt/srt/best",
        "outtmpl": str(subdir / "%(id)s.%(ext)s"),
    })

    def run_sub():
        with yt_dlp.YoutubeDL(sub_opts) as ydl:
            return ydl.extract_info(url, download=True)

    try:
        retry(run_sub, tries=min(args.retries, 3), label="字幕下载")
    except Exception as e:  # noqa: BLE001
        log(f"[subs] 字幕下载失败（{type(e).__name__}）→ 转本地 ASR")
        return info, []

    vid = info.get("id", "video")
    best: list[dict] = []
    for f in sorted(subdir.glob(f"{vid}*")):
        if f.suffix.lower() in (".json", ".json3", ".vtt", ".srt") and "danmaku" not in f.name:
            cues = load_cues(f)
            if cues and len(cues) > len(best):
                best = cues
                log(f"[subs] 命中平台字幕: {f.name}（{len(cues)} 条）")
    return info, best


def download_audio(url: str, args, workdir: Path) -> tuple[dict, Path]:
    import yt_dlp

    adir = workdir / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    opts = ytdlp_opts(args, {
        "format": "bestaudio/best",
        "outtmpl": str(adir / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "0"}],
    })

    def run():
        with yt_dlp.YoutubeDL(opts) as ydl:
            i = ydl.extract_info(url, download=True)
            return i, Path(ydl.prepare_filename(i))

    info, path = retry(run, tries=args.retries, label="音频下载")

    if info.get("entries"):
        info = info["entries"][0]

    if not path.exists():
        cands = [p for p in adir.iterdir() if p.is_file()]
        if not cands:
            raise RuntimeError("音频下载失败：目录为空")
        path = max(cands, key=lambda p: p.stat().st_size)
    return info, path


# ---------------------------------------------------------------- 音频 → 数组
def to_wav16k(src: Path, workdir: Path) -> Path:
    dst = workdir / "audio16k.wav"
    cmd = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg 转码失败：{r.stderr[:500]}")
    return dst


def read_wav_mono16k(path: Path):
    """绕过 PyAV：直接读 PCM。"""
    import numpy as np

    with wave.open(str(path), "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        pcm = w.readframes(n)
    if sr != 16000 or ch != 1:
        raise RuntimeError(f"期望 16k 单声道，实际 {sr}Hz {ch}ch")
    if sw == 2:
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        arr = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"不支持的采样位宽 {sw}")
    return arr


def detect_speech_regions(wav_path: Path, noise_db: float = -35.0, min_silence: float = 0.5,
                          min_speech: float = 0.25, pad: float = 0.15, total: float = 0.0) -> list[tuple[float, float]]:
    """用 ffmpeg silencedetect 找语音区间，替代坏掉的 Silero VAD。

    背景：faster-whisper 1.2.1 自带的 Silero VAD 在本机（onnxruntime 1.28 + Python 3.14）
    输出概率最高只有 0.22（真实语音应 >0.9），导致 vad_filter=True 时把整段音频全过滤、
    返回 0 段。改用 ffmpeg 的能量检测，稳定且无额外依赖。
    """
    cmd = [str(FFMPEG), "-hide_banner", "-nostats", "-i", str(wav_path),
           "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = (r.stderr or "") + (r.stdout or "")

    sil: list[tuple[float, float]] = []
    cur_start = None
    for m in re.finditer(r"silence_(start|end):\s*([-\d.]+)", text):
        kind, val = m.group(1), float(m.group(2))
        if kind == "start":
            cur_start = val
        else:
            sil.append((cur_start if cur_start is not None else 0.0, val))
            cur_start = None
    if cur_start is not None and total:
        sil.append((cur_start, total))

    if not sil:
        return [(0.0, total)] if total else []

    # 静音区间取反 → 语音区间
    speech: list[tuple[float, float]] = []
    prev = 0.0
    for s, e in sil:
        if s - prev >= min_speech:
            speech.append((prev, s))
        prev = max(prev, e)
    if total and total - prev >= min_speech:
        speech.append((prev, total))

    # 加 padding 并合并重叠
    padded = [(max(0.0, s - pad), (e + pad) if total == 0 else min(total, e + pad)) for s, e in speech]
    merged: list[tuple[float, float]] = []
    for s, e in padded:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def audio_duration(path: Path) -> float:
    try:
        r = subprocess.run(
            [str(FFPROBE), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        return float(r.stdout.strip())
    except Exception:
        return 0.0


# ---------------------------------------------------------------- ASR
def resolve_model(name: str) -> str:
    """把模型名解析成本地目录。

    优先用 curl 下好的独立目录（D:\\tools\\v2t\\models\\<repo>），
    避免 huggingface_hub 每次联网校验——huggingface.co 被墙会卡很久。
    """
    if Path(name).is_dir():
        return name
    for cand in (MODELS / name, MODELS / f"faster-whisper-{name}", MODELS / f"models--Systran--faster-whisper-{name}"):
        if cand.is_dir():
            mb = cand / "model.bin"
            if mb.exists() and mb.stat().st_size > 100 * 1024 * 1024:
                log(f"[model] 使用本地模型 {cand}")
                return str(cand)
            if mb.exists():
                log(f"[model] 警告：{mb} 只有 {mb.stat().st_size/1e6:.1f}MB，可能不完整")

    from faster_whisper import download_model

    log(f"[model] 本地没有 {name}，从镜像下载（首次会慢）…")
    return download_model(name, cache_dir=str(MODELS))


_OPENCC = None

# Whisper 会吐出一些罕见/损坏字符，实测样本：
#   U+FE50 小逗号「﹐」、U+FFFD 替换字符「�」、中文里的半角逗号
_PUNCT_FIX = {
    "\ufe50": "，", "\ufe51": "、", "\ufe52": "。", "\ufe54": "；",
    "\ufe55": "：", "\ufe56": "？", "\ufe57": "！",
}


def normalize_text(text: str, chinese: bool = True) -> str:
    """清理 Whisper 输出的异常字符并统一标点。

    - 丢掉 U+FFFD（替换字符，无法还原）
    - 罕见小标点（U+FE50 等）映射为正常全角标点
    - 中文语境下把半角 , ; : ? ! 转全角，读起来像正经文稿
    """
    out = text.replace("\ufffd", "")
    for k, v in _PUNCT_FIX.items():
        out = out.replace(k, v)
    if chinese:
        out = re.sub(r"(?<=[\u4e00-\u9fff]),(?=[\u4e00-\u9fff\s]|$)", "，", out)
        out = re.sub(r"(?<=[\u4e00-\u9fff]);", "；", out)
        out = re.sub(r"(?<=[\u4e00-\u9fff])\?", "？", out)
        out = re.sub(r"(?<=[\u4e00-\u9fff])!", "！", out)
        out = re.sub(r"(?<=[\u4e00-\u9fff]):", "：", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def to_simplified(text: str) -> str:
    """繁体转简体。

    Whisper 中文常输出繁体（实测 medium 模型把「这条路」写成「這條路」），
    而用户要的是简体文案。opencc 缺失时静默跳过，不影响主流程。
    """
    global _OPENCC
    if _OPENCC is None:
        try:
            import opencc
            _OPENCC = opencc.OpenCC("t2s")
        except Exception:
            _OPENCC = False
            log("[text] 未装 opencc，跳过繁转简（pip install opencc-python-reimplemented）")
    if _OPENCC is False:
        return text
    try:
        return _OPENCC.convert(text)
    except Exception:
        return text


def _fold_repetition(text: str, max_repeat: int = 2) -> str:
    """折叠幻觉式重复。

    Qwen3-ASR 在长时间无人声 / 弱语音段（如只有喘气、背景音）会陷入循环，
    实测 43 分钟视频里有 2 段分别把 "Yes." 重复 993 次、499 次，
    单段就能灌进上万字垃圾，把有效内容淹没。

    处理原则：同一句连续重复超过 max_repeat 次，只保留前 max_repeat 次，
    并标注省略数量——不静默丢弃，让用户知道这里模型出过问题。
    """
    if not text:
        return text
    sents = re.split(r"(?<=[.!?。！？])\s*", text)
    sents = [s for s in sents if s is not None]
    out: list[str] = []
    i = 0
    n = len(sents)
    while i < n:
        cur = sents[i]
        j = i + 1
        while j < n and sents[j].strip() == cur.strip():
            j += 1
        run = j - i
        if run > max_repeat and cur.strip():
            out.append(cur.strip())
            out.append(f"[重复{run}次已省略]")
        else:
            out.extend(sents[i:j])
        i = j
    return "".join(out)


def _qwen_transcribe(audio_path: Path, args, dur: float) -> tuple[list[dict], str, float]:
    """走 Qwen3-ASR 引擎（独立 venv 子进程）。

    Qwen3-ASR 中文 CER 约是 Whisper 的一半，抗噪能力强 4 倍（实测见 notes）。
    但它不直接给时间戳，所以按语音区间切块、逐块转写，用块起点当时间戳。
    """
    py = TOOLS / "venv_qwen" / "Scripts" / "python.exe"
    engine = TOOLS / "bin" / "qwen_engine.py"
    if not py.exists():
        raise RuntimeError(f"Qwen 引擎未安装：{py} 不存在")

    regions = [(0.0, dur)]
    if args.vad:
        regions = detect_speech_regions(audio_path, noise_db=args.vad_noise_db,
                                        min_silence=args.vad_min_silence,
                                        pad=args.vad_pad, total=dur) or [(0.0, dur)]
        log(f"[vad] 语音区间 {len(regions)} 段")

    stage("Qwen3-ASR 转写", 35)
    cues: list[dict] = []
    import numpy as np
    audio = read_wav_mono16k(audio_path)

    # 分段与对齐全部交给 qwen_engine.py 内部处理（单进程内循环，
    # 避免每段都重跑一次 28 秒的模型加载）。这里只负责调用与清理文本。
    #
    # 关键：必须显式传 UTF-8 环境变量。WebUI 跑在 pythonw（无控制台）时，
    # 子进程默认按系统 GBK 输出中文，父进程用 utf-8 解码就全变成乱码/丢字。
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [str(py), str(engine), "--audio", str(audio_path), "--json", "--segments"]
    # 语言：用户明确选了就传，auto 则让模型自己判断
    lang_arg = getattr(args, "language", "auto") or "auto"
    cmd += ["--language", str(lang_arg)]
    # 用户在 WebUI / CLI 里选了具体 Qwen 模型时就传下去；
    # 没选（或选的是 whisper 名）则用适配器默认的 Qwen3-ASR-1.7B。
    mname = getattr(args, "model", "") or ""
    if mname and mname.lower().startswith("qwen"):
        cand = MODELS / mname
        if cand.is_dir():
            cmd += ["--model", str(cand)]

    r = subprocess.run(
        cmd,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=3600, env=env)

    data = {}
    for line in reversed((r.stdout or "").strip().splitlines()):
        if line.strip().startswith("{"):
            try:
                data = json.loads(line.strip())
                break
            except Exception:
                continue

    if not data.get("ok"):
        tail = (r.stderr or "")[-600:]
        err = data.get("error", "无输出")
        raise RuntimeError(f"Qwen3-ASR 失败：{err} / {tail}")

    for seg in (data.get("segments") or []):
        txt = (seg.get("text") or "").strip()
        # 清掉 Qwen 残留的语言标记。
        # 坑：模型会把 "language English" 和正文粘在一起，如
        #     "language EnglishWe should order something"
        # 若用 \s+[A-Za-z]+\s* 去匹配，会把正文开头的 "We" 一起吃掉
        # （实测变成 "should order something"，首词丢失）。
        # 所以只匹配已知语言名。
        txt = re.sub(
            r"\blanguage\s+(Chinese|English|Cantonese|Japanese|Korean|Arabic|German|"
            r"French|Spanish|Portuguese|Indonesian|Italian|Russian|Thai|Vietnamese|"
            r"Turkish|Hindi|Malay|Dutch|Swedish|Danish|Finnish|Polish|Czech|"
            r"Filipino|Persian|Greek|Hungarian|Macedonian|Romanian|None)\s*:?\s*",
            "", txt, flags=re.IGNORECASE).strip()
        txt = txt.replace("<asr_text>", "").replace("<|asr_text|>", "").strip()
        if not txt:
            continue
        # 折叠幻觉式重复（Qwen 在弱语音段会循环同一句几百次）
        txt = _fold_repetition(txt)
        # 语言按实际内容判断：中文字符占比 >15% 才当中文处理
        zh = sum(1 for ch in txt if "\u4e00" <= ch <= "\u9fff") > len(txt) * 0.15
        txt = normalize_text(to_simplified(txt) if zh else txt, chinese=zh)
        cues.append({"start": float(seg.get("start", 0.0)),
                     "end": float(seg.get("end", dur)),
                     "text": txt})

    log(f"[qwen] 拿到 {len(cues)} 段文案")
    stage("转写完成", 95)

    if not cues:
        raise RuntimeError("Qwen3-ASR 没有返回任何文本")
    return cues, "zh", dur


def _write_wav16k(path: Path, audio) -> None:
    import numpy as np
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())


def _estimate_noise(audio_path: Path, args) -> tuple[bool, float]:
    """粗估音频是否「嘈杂」，用于 auto 选引擎。

    判定思路（不需要真 VAD）：
      1. 用 ffmpeg silencedetect 静音阈值反推底噪水平；
      2. 再看「语音段占比」——带强 BGM 时几乎没有真正的静音，
         整个时长都会被判成有声，静音段占比会显著偏低。
    返回 (是否嘈杂, 估算底噪 dB)。
    """
    try:
        dur = audio_duration(audio_path) or 0.0
        if dur <= 0:
            return False, -60.0
        regions = detect_speech_regions(
            audio_path,
            noise_db=getattr(args, "vad_noise_db", -35.0),
            min_silence=0.35, pad=0.0, total=dur)
        speech = sum(e - s for s, e in regions) if regions else dur
        ratio = min(1.0, max(0.0, speech / dur))

        # 静音占比低 → 背景声填满了间隙（BGM/环境噪）
        # 经验阈值：正常口播留白约 10~25%，持续 BGM 通常 <5%
        silent_ratio = 1.0 - ratio
        noisy = silent_ratio < 0.05

        # 底噪粗估：用 silencedetect 在更严阈值下是否还能找到静音
        if not noisy:
            strict = detect_speech_regions(
                audio_path, noise_db=getattr(args, "vad_noise_db", -35.0) - 10,
                min_silence=0.35, pad=0.0, total=dur)
            strict_ratio = (sum(e - s for s, e in strict) / dur) if strict else 1.0
            if strict_ratio > 0.95:
                noisy = True    # 再严 10dB 仍无静音 → 底噪很高
        return noisy, -35.0 if noisy else -50.0
    except Exception:
        return False, -50.0


def transcribe(audio_path: Path, args) -> tuple[list[dict], str, float]:
    engine = getattr(args, "engine", "auto")

    # auto：按「语言 + 音频质量」选引擎。
    #
    # 之前的实现只看语言（en→whisper，其他→qwen），这是错的：
    # 「英文」不等于「干净英文朗读」。benchmark 数据是——
    #   LibriSpeech clean：Whisper 1.51 vs Qwen 1.63（Whisper 略胜）
    #   Extreme Noise    ：Whisper 63.17 vs Qwen 16.17（Qwen 好 4 倍）
    # 所以英文只在「音频干净」时才该走 Whisper；带 BGM/嘈杂的英文必须交给 Qwen。
    if engine == "auto":
        lang = (getattr(args, "language", "zh") or "zh").lower()
        noisy, noise_db = _estimate_noise(audio_path, args)
        if lang.startswith("en"):
            if noisy:
                engine = "qwen"
                log(f"[asr] auto 选路：英文 + 音频嘈杂(dB≈{noise_db:.0f}) → qwen 引擎"
                    f"（Whisper 抗噪差 4 倍）")
            else:
                engine = "whisper"
                log(f"[asr] auto 选路：英文 + 音频干净(dB≈{noise_db:.0f}) → whisper 引擎")
        else:
            engine = "qwen"
            log(f"[asr] auto 选路：language={lang} → qwen 引擎")

    if engine == "qwen":
        global _ENGINE_USED
        _ENGINE_USED = "qwen"
        dur = audio_duration(audio_path) or 0.0
        if dur <= 0:
            import numpy as _np
            dur = len(read_wav_mono16k(audio_path)) / 16000.0
        log(f"[asr] 引擎：Qwen3-ASR-1.7B（{dur/60:.1f} 分钟）")
        return _qwen_transcribe(audio_path, args, dur)

    import numpy as np
    from faster_whisper import WhisperModel

    audio = read_wav_mono16k(audio_path)
    dur = len(audio) / 16000.0

    mpath = resolve_model(args.model)
    stage(f"加载模型 {args.model}", 30)
    log(f"[asr] 音频 {dur/60:.1f} 分钟，加载模型 {args.model} ({args.device}/{args.compute_type}) …")
    t0 = time.time()
    model = WhisperModel(mpath, device=args.device, compute_type=args.compute_type,
                         cpu_threads=args.cpu_threads or 0)
    log(f"[asr] 模型就绪 {time.time()-t0:.1f}s")
    stage("转写中", 35)

    # ---- 语音区间检测（自建 VAD，替代坏掉的 Silero）
    if args.vad:
        regions = detect_speech_regions(audio_path, noise_db=args.vad_noise_db,
                                        min_silence=args.vad_min_silence,
                                        pad=args.vad_pad, total=dur)
        speech_secs = sum(e - s for s, e in regions)
        log(f"[vad] 语音区间 {len(regions)} 段，共 {speech_secs/60:.1f} 分钟"
            f"（占比 {speech_secs/dur*100:.0f}%），已剔除静音/纯音乐段")
    else:
        regions = [(0.0, dur)]

    if not regions:
        log("[vad] 未检测到语音，回退为整段转写")
        regions = [(0.0, dur)]

    # ---- 拼接语音区间 + 记录时间偏移，转写后把时间戳映射回原时间轴
    pieces, offsets = [], []
    cursor = 0.0
    for s, e in regions:
        a, b = int(s * 16000), int(e * 16000)
        if b <= a:
            continue
        pieces.append(audio[a:b])
        offsets.append((cursor, s))
        cursor += (b - a) / 16000.0
    merged_audio = np.concatenate(pieces) if pieces else audio
    merged_dur = len(merged_audio) / 16000.0 or 1.0

    def remap(t: float) -> float:
        """把拼接音频上的时间映射回原始时间轴。"""
        for i, (base, orig) in enumerate(offsets):
            nxt = offsets[i + 1][0] if i + 1 < len(offsets) else cursor
            if t < nxt:
                return orig + (t - base)
        return offsets[-1][1] + (t - offsets[-1][0]) if offsets else t

    t0 = time.time()
    lang = None if args.language == "auto" else args.language
    # 中文场景给 initial_prompt 引导简体输出（whisper 默认可能吐繁体）
    init_prompt = None
    if lang and str(lang).startswith("zh"):
        init_prompt = "以下是普通话的句子，请使用简体中文转写。"

    segments, info = model.transcribe(
        merged_audio,
        language=lang,
        beam_size=args.beam_size,
        vad_filter=False,          # 已用自建 VAD 预处理
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        initial_prompt=init_prompt,
    )
    cues: list[dict] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            if lang and str(lang).startswith("zh"):
                text = normalize_text(to_simplified(text), chinese=True)
            else:
                text = normalize_text(text, chinese=False)
            if text:
                cues.append({"start": remap(float(seg.start)), "end": remap(float(seg.end)), "text": text})
        if len(cues) and len(cues) % 25 == 0:
            ratio = min(1.0, cues[-1]["end"] / merged_dur)
            log(f"[asr] 已转写 {ratio*100:.0f}%（{cues[-1]['end']/60:.1f}/{dur/60:.1f} 分钟）",
                pct=35 + 60 * ratio)

    el = time.time() - t0
    speed = (sum(e - s for s, e in regions) / el) if el else 0
    log(f"[asr] 完成：{len(cues)} 段，耗时 {el:.1f}s（{speed:.1f}x 实时）", pct=95)
    return cues, (info.language or "?"), dur


# ---------------------------------------------------------------- 输出
def safe_filename(title: str, maxlen: int = 60) -> str:
    """把平台标题洗成适合做文件名的短标题。

    抖音/快手的标题常带一堆话题标签，原样落盘会得到
    "Vibe知识大赏｜声学研究1年...#抖音前沿科技首发计划 #vibe知识大赏 #ai新星计划..."
    这种又长又乱的名字。这里剥掉 hashtag 与 @提及，只留人话。
    """
    s = str(title or "")
    s = re.sub(r"#\S+", "", s)              # 话题标签
    s = re.sub(r"@\S+", "", s)              # @提及
    s = re.sub(r"https?://\S+", "", s)      # 残留链接
    s = re.sub(r"\s+", " ", s).strip(" ｜|-—_·")
    s = re.sub(r'[\\/:*?"<>|\n\r\t\x00-\x1f]+', "_", s)
    s = s.strip(" ._")[:maxlen].strip(" ._")
    return s or "video"


def hhmmss(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{int(sec//3600):02d}:{int(sec%3600//60):02d}:{int(sec%60):02d}"


def _needs_space(left: str, right: str) -> bool:
    """判断拼接处是否需要补空格。

    中文/日文之间不加空格；拉丁字母或数字相邻时必须加，
    否则 faster-whisper 的分段会粘成 "any other guyI just wanna"。
    """
    if not left or not right:
        return False
    a, b = left[-1], right[0]
    if a.isspace() or b.isspace():
        return False
    # CJK / 全角标点之间不加空格
    cjk = lambda ch: ("\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff"
                      or "\uff00" <= ch <= "\uffef" or "\u3000" <= ch <= "\u303f")
    if cjk(a) or cjk(b):
        return False
    if a in "，。！？；：、）》】”’…—" or b in "，。！？；：、（《【“‘":
        return False
    # 拉丁字母/数字相邻 → 补空格
    return a.isalnum() and b.isalnum()


def _needs_sep(left: str, right: str) -> bool:
    """段落合并时的分隔判断。

    B站原生 AI 字幕是「逐句、无标点」的，直接拼接会粘成一大坨
    （"…学习插件我们想让人工智能…"）。这里在没有标点的地方补一个空格，
    只增加可读性，不篡改原文、不臆造标点。
    """
    if not left or not right:
        return False
    a, b = left[-1], right[0]
    if a.isspace() or b.isspace():
        return False
    punct = "，。！？；：、）》】”’…—,.!?;:)"
    if a in punct or b in punct:
        return False
    cjk = lambda ch: ("\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff"
                      or "\uff00" <= ch <= "\uffef" or "\u3000" <= ch <= "\u303f")
    # 中文句间无标点 → 补空格；拉丁文相邻 → 补空格
    return (cjk(a) and cjk(b)) or (a.isalnum() and b.isalnum())


def join_cues(cues: list[dict]) -> str:
    out = ""
    for c in cues:
        t = c["text"]
        if out and _needs_sep(out, t):
            out += " "
        out += t
    return out


def group_paragraphs(cues: list[dict], gap: float = 1.5, max_len: int = 240,
                     max_dur: float = 30.0) -> list[dict]:
    """把碎句合成段落，便于阅读和检索。

    同时按「静音间隔 / 字数 / 时长」三重切分：只按字数切会让一段长达 100+ 秒，
    用户拿着时间戳回视频里定位答案会很难找。默认一段不超过 30 秒。
    """
    paras: list[dict] = []
    for c in cues:
        if paras:
            p = paras[-1]
            merged = p["text"]
            if _needs_sep(merged, c["text"]):
                merged += " "
            merged += c["text"]
            if (c["start"] - p["end"] <= gap
                    and len(merged) <= max_len
                    and (c["end"] - p["start"]) <= max_dur):
                p["text"] = merged
                p["end"] = c["end"]
                continue
        paras.append({"start": c["start"], "end": c["end"], "text": c["text"]})
    return paras


def write_srt(cues: list[dict], path: Path) -> None:
    def fmt(s: float) -> str:
        s = max(0.0, s)
        return f"{int(s//3600):02d}:{int(s%3600//60):02d}:{int(s%60):02d},{int(s%1*1000):03d}"

    lines = []
    for i, c in enumerate(cues, 1):
        lines.append(f"{i}\n{fmt(c['start'])} --> {fmt(c['end'])}\n{c['text']}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="视频链接 → 文案（字幕优先，无字幕才 ASR）",
        epilog=(
            "B站登录相关命令：\n"
            "  v2t login          扫码登录 B站（获取原生 AI 字幕，有效期约 180 天）\n"
            "  v2t logout         清除本地 B站 登录信息\n"
            "  v2t status         查看登录状态与剩余有效期\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", nargs="?", help="视频 URL 或本地音视频文件")
    ap.add_argument("--outdir", default=None, help="输出目录（默认 <当前目录>/视频文案提取）")
    ap.add_argument("--model", default="large-v3", help="whisper 模型（默认 large-v3）")
    ap.add_argument("--engine", default="auto", choices=["auto", "qwen", "whisper"],
                    help="ASR 引擎：auto(按语言自动选，默认) / qwen(Qwen3-ASR，中文强) / "
                         "whisper(faster-whisper，英文强)")
    ap.add_argument("--language", default="zh", help="语言，zh/en/ja… 或 auto")
    ap.add_argument("--device", default="cuda", help="cuda / cpu")
    ap.add_argument("--compute-type", default="float16", help="float16 / int8_float16 / int8")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--cpu-threads", type=int, default=0, help="CPU 线程数（0=自动）")
    ap.add_argument("--vad", dest="vad", action="store_true", default=True, help="启用自建静音过滤（默认开）")
    ap.add_argument("--no-vad", dest="vad", action="store_false", help="关闭静音过滤")
    ap.add_argument("--vad-noise-db", type=float, default=-35.0, help="静音判定阈值 dB（默认 -35）")
    ap.add_argument("--vad-min-silence", type=float, default=0.5, help="多长静音算一段（秒，默认 0.5）")
    ap.add_argument("--vad-pad", type=float, default=0.15, help="语音段前后留白（秒，默认 0.15）")
    ap.add_argument("--sub-lang", default="zh-Hans", help="优先字幕语言")
    ap.add_argument("--cookies-from-browser", default=None, help="edge/chrome/firefox，抓 B站 AI 字幕需要")
    ap.add_argument("--cookies", default=None, help="cookies.txt 路径")
    ap.add_argument("--proxy", default=None, help="如 http://127.0.0.1:7890")
    ap.add_argument("--retries", type=int, default=4, help="网络重试次数（默认 4）")
    ap.add_argument("--playlist", action="store_true", help="允许整个播放列表")
    ap.add_argument("--no-subs", action="store_true", help="跳过字幕，直接 ASR")
    ap.add_argument("--headed", action="store_true",
                    help="浏览器嗅探时弹出可见窗口（遇到滑块验证可手动过）")
    ap.add_argument("--keep-temp", action="store_true", help="保留中间文件")
    return ap


def run(src: str, args) -> dict:
    """执行一次转写，返回结构化结果。

    args 可以是 argparse.Namespace，也可以是任何带同名属性的对象
    （WebUI 用 SimpleNamespace 调用，不走命令行）。

    返回 dict：
        ok / code / error —— 成败
        md / txt / srt    —— 三个输出文件绝对路径
        title / source / duration / chars / paragraphs / cues / elapsed
    """
    stage("开始", 2)
    is_url = bool(re.match(r"^https?://", src, re.I))
    outdir = Path(args.outdir) if args.outdir else Path.cwd() / "视频文案提取"
    outdir.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    workdir = outdir / "_tmp"
    if workdir.exists() and not args.keep_temp:
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)

    info: dict = {}
    cues: list[dict] = []
    source = "asr"
    audio_dur = 0.0

    # ---- 1) 本地文件直接 ASR
    if not is_url:
        local = Path(src)
        if not local.exists():
            log(f"[err] 文件不存在：{local}")
            return {"ok": False, "code": 2, "error": f"文件不存在：{local}"}
        stage("读取本地文件", 20)
        title = local.stem
        info = {"title": title, "webpage_url": "", "uploader": "", "duration": audio_duration(local)}
    else:
        # ---- 2) URL：先判断是否必须走浏览器通道
        use_browser = browser_fetch is not None and browser_fetch.needs_browser(src)

        if use_browser:
            # 抖音/快手/小红书：yt-dlp 已失效（服务端被判机器，带 cookie 也无效），
            # 直接走浏览器嗅探，不在注定失败的路径上浪费重试时间。
            log("[flow] 该站点需要浏览器嗅探（yt-dlp 对它已失效）")
            stage("浏览器抓取", 10)
            bres = browser_fetch.fetch(
                src, workdir,
                headed=bool(getattr(args, "headed", False)),
            )
            if not bres.get("ok"):
                # 浏览器失败仍给 yt-dlp 一次机会（万一哪天修好了）
                log(f"[browser] 失败：{bres.get('error')} → 降级试 yt-dlp")
            else:
                title = bres.get("title") or "video"
                info = {
                    "title": title,
                    "uploader": bres.get("author") or "",
                    "webpage_url": bres.get("page_url") or src,
                    "duration": (bres.get("duration_ms") or 0) / 1000.0,
                }
                media = Path(bres["media"])
                stage("抽取音轨", 12)
                audio = workdir / "browser_audio.wav"
                ok, err = browser_fetch.extract_audio(media, audio)
                if not ok:
                    log(f"[browser] 抽音轨失败：{err}")
                    bres = {"ok": False, "error": err}
                else:
                    log(f"[browser] 抓取成功：{title[:50]}")
                    source = "browser"

            if not bres.get("ok"):
                # 降级：走原来的 yt-dlp 路径
                use_browser = False

        if not use_browser:
            # ---- 2b) 通用站点：先探字幕
            if not args.no_subs:
                # 2b-0) B站专用快路：直取原生 AI 字幕（需登录 cookie）
                # yt-dlp 调 B站接口不带登录态，会误判"无字幕"→ 白跑 ASR。
                # 这条路径直接带 cookie 调同一接口，秒级且零识别误差。
                if is_url and bili_subs.is_bilibili(src):
                    stage("B站字幕直取", 6)
                    try:
                        br = bili_subs.fetch_bilibili_subtitle(
                            src, prefer_lang=args.sub_lang or "zh")
                        if br.get("ok"):
                            # 统一成内部 cue 格式 {start, end, text}
                            cues = [{"start": float(c["from"]),
                                     "end": float(c["to"]),
                                     "text": c["content"]}
                                    for c in br["cues"]]
                            source = "bilibili_native"
                            info = {
                                "title": br.get("title"),
                                "uploader": br.get("author"),
                                "duration": br.get("duration"),
                                "id": br.get("bvid"),
                            }
                            log(f"[bili] ✓ 原生字幕 {len(cues)} 句"
                                f"（{br.get('lan_doc')}，非 ASR）")
                        else:
                            reason = br.get("reason") or "未知原因"
                            log(f"[bili] ✗ 无法直取字幕：{reason}")
                            # 未登录是最常见的原因，且后果严重（会退回 ASR，术语必错），
                            # 必须明确提示，不能让用户以为工具坏了
                            if "cookie" in reason or "登录" in reason or "失效" in reason:
                                log("")
                                log("  ⚠️  检测到 B站未登录。接下来会退回本地 ASR，")
                                log("      专业术语（如 agent / Harness）很可能识别错误。")
                                log("      建议先执行：  v2t login")
                                log("      登录后可直取 B站官方 AI 字幕，零误差且秒级完成。")
                                log("")
                            info = {"title": br.get("title")} if br.get("title") else {}
                    except Exception as e:  # noqa: BLE001
                        log(f"[bili] 直取异常（继续走通用探测）：{type(e).__name__}: {e}")
                        info = {}

                if not cues:
                    stage("探测平台字幕", 8)
                    try:
                        info2, cues = probe_and_fetch_subs(src, args, workdir)
                        info = info2 or info
                        if cues:
                            source = "subtitle"
                    except Exception as e:
                        log(f"[subs] 探测失败（继续走 ASR）：{type(e).__name__}: {e}")
                        info = info or {}

            if not cues:
                log("[flow] 无可用字幕 → 下载音频跑本地 ASR")
                stage("下载音频", 15)
                info2, audio = download_audio(src, args, workdir)
                info = info2 or info
                title = info.get("title") or "video"
            else:
                title = info.get("title") or "video"

    safe = safe_filename(title)

    # ---- 3) 无字幕才 ASR
    if not cues:
        if is_url:
            if source != "browser":
                pass  # audio 已由 download_audio 下好
            # browser 通道已在上面抽出 audio.wav
        else:
            audio = Path(src)
        wav = to_wav16k(audio, workdir)
        stage("音频转码", 25)
        audio_dur = audio_duration(wav)
        cues, lang, audio_dur = transcribe(wav, args)
    else:
        lang = args.language
        audio_dur = float(info.get("duration") or (cues[-1]["end"] if cues else 0))

    if not cues:
        log("[err] 没拿到任何文案")
        return {"ok": False, "code": 3, "error": "没拿到任何文案（无字幕且 ASR 无输出）"}

    # ---- 中文统一繁转简 + 标点清理（字幕路径也可能拿到繁体字幕）
    # 语言按「文案实际内容」判断，而不是 --language 默认值：
    # 否则英文视频也会被当成中文处理（默认 --language zh）。
    if source == "subtitle":
        sample = "".join(c["text"] for c in cues[:200])
        zh = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff") > len(sample) * 0.15
        lang = "zh" if zh else (info.get("language") or args.language)
        cues = [{**c, "text": normalize_text(to_simplified(c["text"]) if zh else c["text"], chinese=zh)}
                for c in cues if c["text"]]

    # ---- 4) 输出
    stage("整理文案", 96)
    paras = group_paragraphs(cues)
    _SOURCE_LABEL = {
        "bilibili_native": "B站原生 AI 字幕（接口直取，无识别误差）",
        "subtitle": "平台字幕（原样，无识别误差）",
        "browser": f"本地 ASR（{args.model}, {args.device}/{args.compute_type}）",
        "asr": f"本地 ASR（{args.model}, {args.device}/{args.compute_type}）",
    }
    _CHANNEL_LABEL = {
        "bilibili_native": "B站字幕接口（需登录）",
        "subtitle": "yt-dlp",
        "browser": "浏览器嗅探（yt-dlp 对该站失效）",
        "asr": "yt-dlp",
    }
    header = [
        f"# {safe}",
        "",
        f"- 来源：{info.get('webpage_url') or src}",
        f"- UP主/作者：{info.get('uploader') or info.get('channel') or '-'}",
        f"- 时长：{hhmmss(audio_dur)}",
        f"- 文案来源：{_SOURCE_LABEL.get(source, source)}",
        f"- 抓取通道：{_CHANNEL_LABEL.get(source, 'yt-dlp')}",
        f"- 抓取时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 段落数：{len(paras)} / 句数：{len(cues)}",
    ]
    # 原始标题含话题标签，保留下来备查（清洗过的标题放在 H1）
    if title and title != safe:
        header.append(f"- 原始标题：{title}")
    header += [
        "",
        "---",
        "",
    ]

    md_path = outdir / f"{safe}.md"
    txt_path = outdir / f"{safe}.txt"
    srt_path = outdir / f"{safe}.srt"

    md_body = [f"## [{hhmmss(p['start'])}]\n\n{p['text']}\n" for p in paras]
    md_path.write_text("\n".join(header) + "\n".join(md_body), encoding="utf-8")
    txt_path.write_text("\n\n".join(p["text"] for p in paras), encoding="utf-8")
    write_srt(cues, srt_path)

    if not args.keep_temp:
        shutil.rmtree(workdir, ignore_errors=True)

    stage("完成", 100)
    elapsed = time.time() - t_all
    chars = len(txt_path.read_text(encoding="utf-8"))
    log("")
    log(f"✅ 完成，总耗时 {elapsed:.1f}s（来源：{source}）")
    log(f"   md  : {md_path}")
    log(f"   txt : {txt_path}")
    log(f"   srt : {srt_path}")
    log(f"   字数: {chars}")
    return {
        "ok": True,
        "code": 0,
        "title": safe,          # 已清洗：去掉 hashtag/@，可直接当显示名
        "full_title": title,    # 原始平台标题，保留给需要完整信息的地方
        "source": source,
        "md": str(md_path),
        "txt": str(txt_path),
        "srt": str(srt_path),
        "outdir": str(outdir),
        "duration": audio_dur,
        "chars": chars,
        "paragraphs": len(paras),
        "cues": len(cues),
        "language": lang,
        "asr": (None if source == "subtitle"
                else ("Qwen3-ASR-1.7B" if _ENGINE_USED == "qwen"
                      else f"{args.model} / {args.device}/{args.compute_type}")),
        "elapsed": round(elapsed, 1),
    }


def main() -> int:
    args = build_parser().parse_args()

    # ---- B站登录子命令（v2t login / logout / status）
    if args.input in ("login", "logout", "status"):
        try:
            import bili_auth
        except ImportError:
            log("[err] 找不到 bili_auth.py（应与 v2t.py 同目录）")
            return 1

        if args.input == "login":
            r = bili_auth.login()
            if r["ok"]:
                log(f"\n✅ 登录成功，已保存 {r['count']} 条 cookie")
                log(f"   有效期剩余 {r['days_left']} 天")
                log(f"   保存位置：{r['path']}")
                log("")
                log("   登录后，B站视频会优先直取官方 AI 字幕（零识别误差、秒级完成），")
                log("   不再需要下载音频跑本地 ASR。")
                return 0
            log(f"\n❌ {r['reason']}")
            return 1

        if args.input == "logout":
            r = bili_auth.logout()
            if r["removed"]:
                log("✅ 已登出，删除了：")
                for p in r["removed"]:
                    log(f"   {p}")
                log("")
                log("   注意：登出后 B站视频将无法直取字幕，会退回本地 ASR（术语易错）。")
            else:
                log("（本地本来就没有登录信息）")
            return 0

        # status
        print(bili_auth._fmt_status(bili_auth.status()))
        return 0

    if not args.input:
        build_parser().print_help()
        return 1

    try:
        res = run(args.input, args)
    except KeyboardInterrupt:
        log("\n[中断] 用户取消")
        return 130
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        log(f"\n[err] {type(e).__name__}: {e}")
        return 1
    return int(res.get("code", 0))


if __name__ == "__main__":
    sys.exit(main())
