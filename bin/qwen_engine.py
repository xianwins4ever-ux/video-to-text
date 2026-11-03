# -*- coding: utf-8 -*-
"""Qwen3-ASR 引擎适配器（独立 venv 子进程）。

为什么要独立进程
----------------
Qwen3-ASR 依赖 torch + transformers，装在 D:\\tools\\v2t\\venv_qwen；
而 v2t 主程序跑在系统 Python 3.14（faster-whisper + ctranslate2）。
两套依赖互不相容（一个有 torch 没 ctranslate2，另一个相反），
硬塞一起会打架。所以用子进程隔离：主程序调本脚本，结果走 stdout JSON。

踩过的坑（都在这里修掉了，别再踩一遍）
--------------------------------------
1. **权重键名前缀不同**：官方 checkpoint 存的是 `thinker.audio_tower.*` /
   `thinker.model.*`，而 transformers 的 Qwen3ASRForConditionalGeneration
   期望 `model.audio_tower.*` / `model.language_model.*`。
   已在安装阶段生成 `model_fixed.safetensors` 做过重映射。
2. **audio_config.output_dim 被错误展平**：config.json 里 thinker_config.audio_config
   的 output_dim 是 2048，但 from_pretrained 后会变成 3584（取了错的层级）。
   必须手动 `cfg.audio_config.output_dim = cfg.text_config.hidden_size`。
3. **mask 语义**：input_features_mask 形状是 (batch, padded_feature_length)，
   对 3000 帧 mel 而言 encoder 实际输出 390 帧特征。
4. **audio_pad 要按真实特征数展开**：Qwen3ASR 要求 input_ids 里
   `<|audio_pad|>` 的个数 == audio_features 的帧数。先跑一次 encoder 拿
   n_feat，再把 prompt 里的占位符展开成 n_feat 个。
5. 帧数在 `pooler_output.shape[0]`，不是 shape[1]（shape 是 (n_feat, hidden)）。

用法
----
    <venv_python> qwen_engine.py --audio x.wav --model <dir> --json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

TOOLS = Path(r"D:\tools\v2t")
FFMPEG_DIR = Path(r"D:\hyperframes-tools\ffmpeg")
FFMPEG = FFMPEG_DIR / "ffmpeg.exe"
DEFAULT_MODEL = TOOLS / "models" / "Qwen3-ASR-1.7B"

os.environ.setdefault("HF_HOME", str(TOOLS / "models"))

# 强制 UTF-8 输出。Windows 下若没有控制台（pythonw）或未设 PYTHONIOENCODING，
# Python 会按系统 GBK 写 stdout，父进程按 utf-8 读就会得到乱码/丢字。
# 这个坑在「命令行手测正常、WebUI 里全是乱码」时极难定位，必须写死。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def log(msg: str) -> None:
    print(f"[qwen] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- ASR
class QwenASR:
    def __init__(self, model_dir: Path, device: str = "cuda"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.model = None
        self.proc = None
        self.tok = None
        self.audio_pad_id = None

    def load(self) -> None:
        if self.model is not None:
            return
        import warnings
        warnings.filterwarnings("ignore")
        import gc
        import torch
        from transformers import (
            Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor, Qwen3ASRConfig)

        cfg = Qwen3ASRConfig.from_pretrained(str(self.model_dir))
        # 坑 2：output_dim 修正
        cfg.audio_config.output_dim = cfg.text_config.hidden_size

        # 坑 8（2026-11-02，真实素材上才暴露）：内存/显存双峰。
        # 原来的写法是 load_file() 把 4.4GB 权重全读进 CPU，再 .to("cuda")
        # 复制一份到 GPU —— 峰值要 2×4.4GB，本机（可用 8.4GB 内存、AE 占走
        # 3.4GB 显存后只剩 4.5GB）直接 OOM，进程被静默杀掉，连报错都没有。
        #
        # 改法：用 meta device 建空壳模型，再逐个张量「读一个、搬一个、放一个」，
        # CPU 侧全程只驻留单个张量。峰值从 2×4.4GB 降到 ~4.4GB（只在 GPU 上）。
        fixed = self.model_dir / "model_fixed.safetensors"
        if fixed.exists():
            from safetensors.torch import load_file
            _preflight(_estimate_weight_bytes(fixed), self.device)

            # 直接走「普通构建 + load_state_dict + 转 fp16 + 上设备」这条最朴素的路。
            # 试过两种省内存优化，都不行：
            #   ① 逐张量 copy_ 到 to_empty 模型 → 能跑但输出退化成 "a a a we see"，
            #      且不报错，比崩溃更难查；
            #   ② meta + assign=True → buffer 仍是 meta，直接抛 NotImplementedError。
            # 结论：权重加载这一步不要抖机灵，内存不够就明确报错让用户关程序，
            # 而不是用奇技淫巧换一个「跑得通但结果是垃圾」的状态。
            model = Qwen3ASRForConditionalGeneration(cfg)
            sd = load_file(str(fixed), device="cpu")
            res = model.load_state_dict(sd, strict=False)
            if res.missing_keys:
                log(f"警告：缺失 {len(res.missing_keys)} 个张量（{res.missing_keys[:2]}）")
            if res.unexpected_keys:
                log(f"警告：多余 {len(res.unexpected_keys)} 个张量")
            del sd
            gc.collect()
            model = model.half().to(self.device).eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(f"权重加载完成（{self.device}/float16）")
        else:
            log("未找到 model_fixed.safetensors，回退 from_pretrained（质量可能受损）")
            model = Qwen3ASRForConditionalGeneration.from_pretrained(
                str(self.model_dir), dtype=torch.float16,
                device_map=self.device, low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True)

        self.model = model.eval()
        self.proc = Qwen3ASRProcessor.from_pretrained(str(self.model_dir))
        self.tok = self.proc.tokenizer
        # encoder 要求时间维是 n_window*2 的整数倍
        self.chunk_len = int(getattr(cfg.audio_config, "n_window", 50)) * 2

        raw = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.audio_pad_id = raw.get("thinker_config", {}).get("audio_token_id", 151676)

    def transcribe_array(self, audio, sr: int = 16000,
                         language: str | None = None) -> str:
        """audio: float32 numpy，16kHz 单声道。

        language 传模型认识的英文名（Chinese / English …）或 None 让它自判。
        """
        import torch
        self.load()

        # 坑 6（最隐蔽，2026-11-02 排查出来的）：
        # WhisperFeatureExtractor 默认 padding='max_length' 且 chunk_length=30，
        # 会把**任意长度**音频都补到 T=3000 帧。于是短片段变成「绝大部分是静音
        # 填充」，模型直接摆烂输出 "language None" 而没有任何报错。
        # 表现：同一段音频切碎后只有第一块能识别，其余全空——极难定位。
        # 修法：显式 padding=False, truncation=False，让 T 随真实时长变化。
        feat = self.proc.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt",
            return_attention_mask=True, padding=False, truncation=False)
        mel = feat["input_features"].to(self.device).half()

        # 坑 7：encoder 要求 padded_feature_length 是 (n_window*2)=100 的整数倍。
        # 任意时长会被截断成非整数倍（如 2920），必须补零到下一个整数倍。
        import torch as _t
        B, M, T = mel.shape
        ck = self.chunk_len
        if T % ck != 0:
            pad = ck - (T % ck)
            mel = _t.cat([mel, mel.new_zeros((B, M, pad))], dim=2)
            T = mel.shape[2]
        mask = torch.ones((B, T), dtype=torch.long, device=self.device)

        # 坑 4/5：先算真实特征帧数
        with torch.no_grad():
            enc = self.model.get_audio_features(mel, mask, return_dict=True)
        n_feat = enc.pooler_output.shape[0]

        # 语言指定：Qwen3-ASR 的 prompt 里 system 段可放语言指令，
        # 官方做法是在 assistant 起始后吐出 "language xxx<asr_text>"。
        # 这里用 system 提示更稳（指定语言能显著降低小语种误判）。
        sys_prompt = f"Transcribe the speech in {language}." if language else ""
        prompt = (f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                  "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
                  "<|im_start|>assistant\n")
        ids_list = self.tok(prompt, add_special_tokens=False).input_ids
        expanded = []
        for x in ids_list:
            expanded.extend([self.audio_pad_id] * n_feat if x == self.audio_pad_id else [x])
        ids = torch.tensor([expanded], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model.generate(input_ids=ids, input_features=mel,
                                      input_features_mask=mask, max_new_tokens=2048)
        text = self.tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0]
        return _clean(text)


def _estimate_weight_bytes(path: Path) -> int:
    """按文件大小估算 fp16 权重占用的显存（≈ 文件字节数）。"""
    try:
        return path.stat().st_size
    except Exception:
        return 0


def _preflight(weight_bytes: int, device: str) -> None:
    """加载前预检资源，把「静默被杀」变成可读的错误信息。"""
    import ctypes
    GiB = 1024 ** 3

    if device == "cuda":
        try:
            import torch
            free_b, total_b = torch.cuda.mem_get_info()
        except Exception:
            free_b, total_b = 0, 0
        headroom = int(1.2 * GiB)
        if total_b and weight_bytes + headroom > free_b:
            # 列出占用显存的进程，帮用户定位该关什么
            holders = ""
            try:
                import subprocess
                r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                    "--format=csv,noheader"], capture_output=True, text=True, timeout=8)
                lines = [l.strip() for l in (r.stdout or "").splitlines() if l.strip()]
                if lines:
                    holders = " 占用显存的进程：" + "；".join(lines[:4])
            except Exception:
                pass
            raise RuntimeError(
                f"显存不足，无法加载模型。\n"
                f"  模型需要约 {weight_bytes/GiB:.1f} GB，另需 {headroom/GiB:.1f} GB 推理余量\n"
                f"  当前可用 {free_b/GiB:.1f} GB / 共 {total_b/GiB:.1f} GB{holders}\n"
                f"  处理办法：① 关闭占显存的程序（After Effects / 游戏 / 其他 AI 任务）后重试；"
                f"② 或改用 Qwen3-ASR-0.6B（约 1.2 GB）")

    # 系统内存：加载时至少要能容纳权重 + 一点余量
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            need = weight_bytes * 1.3
            if need > m.ullAvailPhys:
                raise RuntimeError(
                    f"内存不足，无法加载模型。\n"
                    f"  需要约 {need/GiB:.1f} GB，当前可用物理内存 {m.ullAvailPhys/GiB:.1f} GB\n"
                    f"  处理办法：关闭占用内存的程序（After Effects 等）后重试，"
                    f"或改用 Qwen3-ASR-0.6B")
    except RuntimeError:
        raise
    except Exception:
        pass


def _clean(text: str) -> str:
    """去掉 <asr_text> 之类残留标记。"""
    t = text.strip()
    for mark in ("language Chinese", "<asr_text>", "<|asr_text|>"):
        t = t.replace(mark, "")
    return t.strip()


# 代码 → 模型认识的英文语言名（config.json 里 support_languages 的取值）
_LANG_MAP = {
    "zh": "Chinese", "zh-cn": "Chinese", "zh-hans": "Chinese", "cmn": "Chinese",
    "zh-tw": "Chinese", "zh-hant": "Chinese",
    "yue": "Cantonese", "cantonese": "Cantonese",
    "en": "English", "en-us": "English", "en-gb": "English",
    "ja": "Japanese", "jp": "Japanese",
    "ko": "Korean",
    "ar": "Arabic", "de": "German", "fr": "French", "es": "Spanish",
    "pt": "Portuguese", "id": "Indonesian", "it": "Italian", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "tr": "Turkish", "hi": "Hindi",
    "ms": "Malay", "nl": "Dutch", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "pl": "Polish", "cs": "Czech", "fil": "Filipino",
    "fa": "Persian", "el": "Greek", "hu": "Hungarian", "mk": "Macedonian",
    "ro": "Romanian",
}


def resolve_language(code: str) -> str | None:
    """把 UI/CLI 的语言代码转成模型认识的名字；auto 或未知返回 None（让模型自判）。"""
    c = (code or "").strip().lower()
    if not c or c == "auto":
        return None
    return _LANG_MAP.get(c, c.capitalize() if c.isalpha() else None)


# ---------------------------------------------------------------- 音频工具
def read_wav16k(path: Path):
    import numpy as np
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        pcm = w.readframes(w.getnframes())
    if sr != 16000 or ch != 1:
        raise RuntimeError(f"需要 16k 单声道，实际 {sr}Hz {ch}ch")
    return np.frombuffer(pcm, dtype=np.int16).astype("float32") / 32768.0


def duration_of(path: Path) -> float:
    try:
        r = subprocess.run(
            [str(FFMPEG_DIR / "ffprobe.exe"), "-v", "error", "-show_entries",
             "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3-ASR 适配器")
    ap.add_argument("--audio", required=True, help="16k 单声道 wav")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", action="store_true", help="兼容参数，始终输出 JSON")
    ap.add_argument("--segments", action="store_true",
                    help="按 28s 窗口分段转写，输出带时间戳的 segments")
    ap.add_argument("--win", type=float, default=28.0, help="分段窗口秒数")
    ap.add_argument("--language", default="auto",
                    help="语言：auto(模型自判) / zh / en / ja / yue … 会映射成模型认识的英文名")
    a = ap.parse_args()

    wav = Path(a.audio)
    if not wav.exists():
        print(json.dumps({"ok": False, "error": f"音频不存在：{wav}"}, ensure_ascii=False))
        return 2

    try:
        import time
        import numpy as np
        audio = read_wav16k(wav)
        dur = len(audio) / 16000.0
        eng = QwenASR(Path(a.model), a.device)

        t_load = time.time()
        eng.load()
        log(f"模型加载 {time.time()-t_load:.1f}s")

        # 无论是否 --segments，都走分段路径。
        # 2026-11-02 实测：整段直喂在 >30s 音频上会生成退化乱码（70s 就崩），
        # 所以「整段直喂」这条路彻底删掉，短音频也当作一段处理。
        WIN = a.win
        slices, cur = [], 0.0
        while cur < dur - 0.5:
            nxt = min(cur + WIN, dur)
            if nxt < dur:
                lo = int(max(cur + WIN * 0.6, nxt - 2.5) * 16000)
                hi = int(min(nxt + 2.5, dur) * 16000)
                if hi > lo:
                    seg = audio[lo:hi]
                    hop, best, best_e = 1600, nxt, 1e9
                    for off in range(0, len(seg) - hop, hop):
                        e = float(np.abs(seg[off:off + hop]).mean())
                        if e < best_e:
                            best_e, best = e, lo / 16000.0 + off / 16000.0
                    nxt = max(cur + WIN * 0.5, min(best, nxt + 2.5))
            slices.append((cur, nxt))
            cur = nxt

        log(f"切成 {len(slices)} 段（每段 ≤{WIN:.0f}s）")
        lang_name = resolve_language(a.language)
        if lang_name:
            log(f"指定语言：{a.language} → {lang_name}")
        else:
            log("语言：自动判断")
        segments = []
        t_all = time.time()
        for i, (s, e) in enumerate(slices, 1):
            piece = audio[int(s * 16000):int(e * 16000)]
            if len(piece) < 1600:
                continue
            txt = eng.transcribe_array(piece, 16000, language=lang_name)
            if txt:
                segments.append({"start": round(s, 2), "end": round(e, 2), "text": txt})
            log(f"  [{i}/{len(slices)}] {s:.0f}-{e:.0f}s -> {txt[:50]}")

        el = time.time() - t_all
        full = "".join(x["text"] for x in segments)
        log(f"转写完成 {el:.1f}s（{dur/max(el,0.01):.1f}x 实时）")
        print(json.dumps({"ok": True, "text": full,
                          "language": lang_name or "auto",
                          "duration": dur,
                          "elapsed": round(el, 1), "segments": segments},
                         ensure_ascii=False))
        return 0
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"},
                         ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
