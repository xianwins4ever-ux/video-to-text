#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v2t WebUI —— 浏览器里的「视频链接 → 文案」工作台

设计取舍：
- 零新增依赖：只用标准库 http.server，不装 flask/fastapi（C 盘只剩几个 GB，
  且这个工具的价值在于「随时能开」，不该因为缺包开不起来）。
- 单实例 + 单 worker：GPU 显存只有 8GB，两个任务并行会互相抢显存直接 OOM。
  所以任务走串行队列，提交多个也一个一个跑。
- 只监听 127.0.0.1，并校验 token：防止本机任意网页 POST 到本服务（CSRF）。

用法：
    python webui.py                 # 起在 7788，自动开浏览器
    python webui.py --port 8899     # 指定端口
    python webui.py --no-browser    # 不自动开浏览器
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

TOOLS = Path(r"D:\tools\v2t")
DATA = TOOLS / "data"
DATA.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = DATA / "uploads"

# pythonw.exe 无控制台时 sys.stdout / sys.stderr 都是 None，
# v2t.py 里的 print(..., file=sys.stderr) 会直接抛 AttributeError。
# 抢在 import v2t 之前把三件套换成日志文件。
_LOG_PATH = DATA / "webui.log"
if sys.stdout is None or sys.stderr is None:
    _log = open(_LOG_PATH, "a", encoding="utf-8", errors="replace", buffering=1)
    if sys.stdout is None:
        sys.stdout = _log
    if sys.stderr is None:
        sys.stderr = _log

sys.path.insert(0, str(TOOLS / "bin"))
import v2t  # noqa: E402

try:
    import bili_auth  # noqa: E402
except Exception:  # noqa: BLE001
    bili_auth = None

VERSION = "2.0"
DEFAULT_PORT = 7788
CONFIG_PATH = DATA / "config.json"
HISTORY_PATH = DATA / "history.json"

TOKEN = secrets.token_urlsafe(24)


# ------------------------------------------------------------------ 配置
def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def default_outdir() -> str:
    cfg = load_config()
    out = cfg.get("outdir")
    # 启动时把记下的输出目录也纳入白名单，否则重启后老记录下载会 403
    allow_root(out)
    return out or r"D:\素材\视频文案提取"


# ------------------------------------------------------------------ 模型探测
def list_models() -> list[dict]:
    """扫描本地已下载的 whisper 模型（不走网络，秒回）。"""
    out: list[dict] = []
    if not v2t.MODELS.is_dir():
        return out
    for d in sorted(v2t.MODELS.iterdir()):
        if not d.is_dir():
            continue
        mb = d / "model.bin"
        if not mb.exists() or mb.stat().st_size < 50 * 1024 * 1024:
            continue
        name = d.name
        for pre in ("models--Systran--faster-whisper-", "faster-whisper-"):
            if name.startswith(pre):
                name = name[len(pre):]
                break
        out.append({"name": name, "size_mb": round(mb.stat().st_size / 1048576)})
    # large > medium > small 排序
    order = {"large-v3": 0, "large-v2": 1, "medium": 2, "small": 3, "base": 4, "tiny": 5}
    out.sort(key=lambda m: (order.get(m["name"], 9), m["name"]))
    return out


def list_qwen_models() -> list[dict]:
    """扫描本地 Qwen3-ASR 模型。

    判定标准：目录里有 model_fixed.safetensors（我们重映射过键名的那份），
    或原始 config.json 里 architectures 含 Qwen3ASR。
    只看总体积，不逐文件遍历——模型动辄几 GB。
    """
    out: list[dict] = []
    if not v2t.MODELS.is_dir():
        return out
    for d in sorted(v2t.MODELS.iterdir()):
        if not d.is_dir() or not d.name.lower().startswith("qwen3-asr"):
            continue
        cfgs = [d / "config.json"]
        is_qwen = (d / "model_fixed.safetensors").exists()
        if not is_qwen:
            for c in cfgs:
                try:
                    txt = c.read_text(encoding="utf-8")
                    if "Qwen3ASR" in txt:
                        is_qwen = True
                        break
                except Exception:
                    pass
        if not is_qwen:
            continue
        # 体积：优先算权重文件
        total = 0
        for pat in ("model_fixed.safetensors", "model-*.safetensors", "*.bin"):
            files = list(d.glob(pat))
            if files:
                total = sum(f.stat().st_size for f in files if f.is_file())
                break
        out.append({"name": d.name, "size_mb": round(total / 1048576)})
    out.sort(key=lambda m: m["name"])
    return out


def gpu_info() -> dict:
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        return {"available": n > 0, "count": n}
    except Exception as e:
        return {"available": False, "count": 0, "error": str(e)}


# ------------------------------------------------------------------ 任务
class Job:
    """一次提交（可含多个视频源）。事件流供 SSE 消费。"""

    def __init__(self, jid: str, sources: list[str], opts: dict):
        self.id = jid
        self.sources = sources
        self.opts = opts
        self.status = "queued"          # queued | running | done | error
        self.pct = 0.0
        self.stage = "排队中"
        self.results: list[dict] = []
        self.error: str | None = None
        self.created = time.time()
        self.events: list[dict] = []
        self.cond = threading.Condition()

    def push(self, ev: dict) -> None:
        with self.cond:
            self.events.append(ev)
            self.cond.notify_all()

    def finish(self, status: str, error: str | None = None) -> None:
        with self.cond:
            self.status = status
            self.error = error
            if status == "done":
                self.pct, self.stage = 100.0, "完成"
            self.events.append({
                "type": "end", "status": status, "error": error,
                "results": self.results, "pct": self.pct,
            })
            self.cond.notify_all()


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
WORK_QUEUE: "queue.Queue[Job]" = queue.Queue()

# B站登录：扫码要开浏览器，会阻塞几十秒，所以放后台线程跑，前端轮询状态。
BILI_LOGIN_STATE: dict = {"running": False, "message": "", "started": 0.0}
BILI_LOGIN_LOCK = threading.Lock()

# /api/file 的白名单：只允许读这些根目录下的文件，避免变成任意文件读取接口。
# v2t 工具目录 + 任何用户选过的输出目录（每跑一次就记一个）。
ALLOWED_ROOTS: set[str] = {str(TOOLS.resolve())}


def allow_root(path: str | None) -> None:
    if not path:
        return
    try:
        ALLOWED_ROOTS.add(str(Path(path).resolve()))
    except Exception:
        pass


def _job_opts(payload: dict) -> SimpleNamespace:
    """把前端提交的 JSON 变成 v2t.run() 认识的对象（属性名与 CLI 一致）。"""
    device = payload.get("device") or "cuda"
    if device == "cuda" and not gpu_info().get("available"):
        device = "cpu"
    compute = payload.get("compute_type") or ("float16" if device == "cuda" else "int8")
    return SimpleNamespace(
        outdir=payload.get("outdir") or default_outdir(),
        model=payload.get("model") or "large-v3",
        engine=payload.get("engine") or "auto",
        language=payload.get("language") or "zh",
        device=device,
        compute_type=compute,
        beam_size=int(payload.get("beam_size") or 5),
        cpu_threads=int(payload.get("cpu_threads") or 0),
        vad=bool(payload.get("vad", True)),
        vad_noise_db=float(payload.get("vad_noise_db", -35.0)),
        vad_min_silence=float(payload.get("vad_min_silence", 0.5)),
        vad_pad=float(payload.get("vad_pad", 0.15)),
        sub_lang=payload.get("sub_lang") or "zh-Hans",
        cookies_from_browser=payload.get("cookies_from_browser") or None,
        cookies=payload.get("cookies") or None,
        proxy=payload.get("proxy") or None,
        retries=int(payload.get("retries") or 4),
        playlist=False,
        no_subs=bool(payload.get("no_subs", False)),
        headed=bool(payload.get("headed", False)),
        keep_temp=False,
    )


def worker() -> None:
    while True:
        job = WORK_QUEUE.get()
        try:
            _run_job(job)
        except Exception as e:  # noqa: BLE001
            job.push({"type": "log", "msg": f"[err] {type(e).__name__}: {e}"})
            job.finish("error", str(e))
        finally:
            WORK_QUEUE.task_done()


def _run_job(job: Job) -> None:
    total = len(job.sources)
    job.status = "running"
    job.push({"type": "start", "total": total, "sources": job.sources})
    allow_root(job.opts.get("outdir"))

    for idx, src in enumerate(job.sources):
        base = idx / total * 100.0
        span = 100.0 / total
        job.push({"type": "item", "index": idx, "total": total, "src": src})

        # 每个源把 v2t 的日志接出来，并按该源的百分比折算到总进度
        def emitter(ev, _base=base, _span=span, _idx=idx):
            if ev.get("type") == "progress" and ev.get("pct") is not None:
                p = _base + _span * (float(ev["pct"]) / 100.0)
                job.pct = round(p, 1)
                job.push({"type": "progress", "pct": round(p, 1), "msg": ev.get("msg", "")})
            elif ev.get("type") == "stage":
                job.stage = f"[{_idx + 1}/{total}] {ev.get('stage', '')}"
                if ev.get("pct") is not None:
                    job.pct = round(_base + _span * (float(ev["pct"]) / 100.0), 1)
                job.push({"type": "stage", "stage": job.stage, "pct": job.pct})
            else:
                job.push({"type": "log", "msg": ev.get("msg", "")})

        v2t.set_emitter(emitter)
        try:
            res = v2t.run(src, _job_opts(job.opts))
            job.results.append(res)
            job.push({"type": "item_done", "index": idx, "result": res})
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            res = {"ok": False, "code": 1, "error": f"{type(e).__name__}: {e}", "src": src}
            job.results.append(res)
            job.push({"type": "item_done", "index": idx, "result": res})
        finally:
            v2t.set_emitter(None)

    ok = any(r.get("ok") for r in job.results)
    # 上传的临时拷贝：转成功就删（md/txt/srt 已经产出，源文件没必要留着占 D 盘）。
    # 失败则保留，方便用户/agent 拿原文件复现问题。
    if ok:
        for s in job.sources:
            try:
                p = Path(s).resolve()
                if p.exists() and UPLOAD_DIR.resolve() in p.parents:
                    p.unlink()
                    # 顺便清掉空的子目录
                    try:
                        p.parent.rmdir()
                    except OSError:
                        pass
            except Exception:
                pass
    hist = load_history()
    for r in job.results:
        if r.get("ok"):
            hist.insert(0, {
                "title": r.get("title"), "md": r.get("md"), "txt": r.get("txt"),
                "srt": r.get("srt"), "source": r.get("source"),
                "elapsed": r.get("elapsed"), "time": time.strftime("%Y-%m-%d %H:%M"),
            })
    save_history(hist[:60])
    job.finish("done" if ok else "error", None if ok else "全部失败")


def load_history() -> list[dict]:
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(items: list[dict]) -> None:
    try:
        HISTORY_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


# ------------------------------------------------------------------ HTML
HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v2t · 视频转文案</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><rect width='64' height='64' rx='14' fill='%230d1117'/><path d='M24 18l22 14-22 14z' fill='%232ee6a8'/><rect x='14' y='46' width='36' height='4' rx='2' fill='%234b9eff'/></svg>">
<style>
/* ==========================================================================
   设计系统
   字体：Inter(拉丁) + Noto Sans SC(中文·思源黑体) + Maple Mono NF CN(等宽)
         —— 三者本机均已安装。原字体栈的 -apple-system 在 Windows 无效，
            中文实际落到微软雅黑，深色背景下小字号笔画发虚，这是"难看"的主因。
   字号：11 / 12 / 13 / 14 / 15 / 17 / 20，字重 400·500·600
   间距：8px 网格   圆角：6/9/12/16   缓动：cubic-bezier(.23,1,.32,1)
   ========================================================================== */
:root{
  --bg:#0b0e12;
  --surface:#111419;
  --surface-2:#0e1116;
  --surface-3:#171b21;
  --surface-4:#1c2129;
  --border:#232830;
  --border-2:#2e343d;
  --text:#e8ebef;
  --text-2:#9aa4b2;
  --text-3:#6b7480;
  --accent:#3ddc97;
  --accent-2:#2bc484;
  --accent-soft:rgba(61,220,151,.12);
  --blue:#5b9dff;
  --blue-soft:rgba(91,157,255,.12);
  --red:#ff7b72;
  --amber:#f0b429;
  --r-sm:6px; --r-md:9px; --r-lg:12px; --r-xl:16px;
  --sh-sm:0 1px 2px rgba(0,0,0,.35);
  --sh-md:0 4px 14px rgba(0,0,0,.4);
  --sh-lg:0 16px 44px rgba(0,0,0,.55);
  --ease-out:cubic-bezier(.23,1,.32,1);
  --ease-io:cubic-bezier(.77,0,.175,1);
  --font:"Inter","Noto Sans SC",-apple-system,"Segoe UI",system-ui,sans-serif;
  --mono:"Maple Mono NF CN","Cascadia Code",Consolas,ui-monospace,monospace;
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0; background:var(--bg); color:var(--text);
  font-family:var(--font); font-size:14px; line-height:1.55;
  font-feature-settings:"cv05" 1,"ss01" 1;  /* Inter: 更好分辨的 l / 1 */
  display:flex; flex-direction:column; overflow:hidden;
  -webkit-font-smoothing:antialiased;
}

/* ---------- 顶栏：半透明材质，内容可从下方穿过 ---------- */
header{
  display:flex; align-items:center; gap:12px; padding:13px 22px;
  background:rgba(17,20,25,.82);
  backdrop-filter:blur(18px) saturate(160%);
  -webkit-backdrop-filter:blur(18px) saturate(160%);
  border-bottom:1px solid var(--border);
  flex:0 0 auto; z-index:10;
}
.logo{
  width:32px;height:32px;border-radius:9px;
  background:linear-gradient(150deg,#14402f,#0c1b2a);
  display:grid;place-items:center;
  border:1px solid rgba(61,220,151,.18);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.06), var(--sh-sm);
}
h1{
  font-size:15px; margin:0; font-weight:600;
  letter-spacing:-.01em;              /* 字号越大 tracking 越紧 */
  display:flex; align-items:baseline; gap:9px;
}
h1 span{
  color:var(--text-3); font-weight:400; font-size:12px;
  letter-spacing:.01em;               /* 小字略微放宽 */
}
.spacer{flex:1}
.dot{
  width:7px;height:7px;border-radius:50%;background:var(--accent);
  box-shadow:0 0 0 3px var(--accent-soft), 0 0 10px rgba(61,220,151,.5);
  display:inline-block;vertical-align:middle;
  transition:background 200ms var(--ease-out), box-shadow 200ms var(--ease-out);
}
.ver{
  font-size:11px; color:var(--text-3); font-weight:500;
  letter-spacing:.04em; text-transform:uppercase;
  border:1px solid var(--border); padding:3px 9px; border-radius:20px;
}
/* B站登录状态按钮 */
.bili-btn{
  display:inline-flex;align-items:center;gap:6px;
  font-size:11.5px;font-weight:500;
  padding:4px 10px;border-radius:20px;cursor:pointer;
  border:1px solid var(--border);background:transparent;color:var(--text-2);
  transition:all 160ms var(--ease-out);white-space:nowrap;
}
.bili-btn:hover{border-color:var(--accent);color:var(--accent)}
.bili-btn.on{border-color:var(--accent);color:var(--accent);
  background:var(--accent-soft)}
.bili-btn .b-dot{width:6px;height:6px;border-radius:50%;background:var(--text-3);flex:none}
.bili-btn.on .b-dot{background:var(--accent);box-shadow:0 0 6px var(--accent)}

/* ---------- 主区 ---------- */
main{
  flex:1;display:grid;grid-template-columns:var(--aside-w,470px) 7px 1fr;
  overflow:hidden;min-height:0;
}
.splitter{cursor:col-resize;position:relative;background:transparent;flex:0 0 auto}
.splitter:after{
  content:"";position:absolute;inset:0 3px;background:var(--border);
  border-radius:2px;transition:background 160ms var(--ease-out);
}
.splitter:hover:after,.splitter.dragging:after{background:var(--accent)}
body.resizing{cursor:col-resize;user-select:none}
body.resizing *{pointer-events:none}

aside{
  background:var(--surface-2);
  padding:20px 22px 26px; overflow-y:auto;
  display:flex; flex-direction:column; gap:18px;
  border-right:1px solid var(--border); min-width:0;
}
.out{display:flex;flex-direction:column;overflow:hidden;min-height:0}
.group{display:flex;flex-direction:column;gap:8px}

/* ---------- 标签：小字 + 正 tracking（apple §15） ---------- */
label.lbl{
  font-size:11.5px; color:var(--text-3); font-weight:500;
  letter-spacing:.03em;
  display:flex; justify-content:space-between; align-items:center;
}
label.lbl b{color:var(--accent);font-weight:500;letter-spacing:.02em}

/* ---------- 输入控件 ---------- */
textarea,input,select{
  width:100%; background:#090c10; border:1px solid var(--border);
  color:var(--text); border-radius:var(--r-md);
  padding:10px 13px; font-family:var(--font); font-size:13.5px; line-height:1.6;
  outline:none; box-shadow:inset 0 1px 2px rgba(0,0,0,.25);
  transition:border-color 160ms var(--ease-out), box-shadow 160ms var(--ease-out),
             background 160ms var(--ease-out);
}
textarea::placeholder,input::placeholder{color:#4d5560}
textarea:hover,input:hover,select:hover{border-color:var(--border-2)}
textarea:focus,input:focus,select:focus{
  border-color:rgba(61,220,151,.5); background:#0a0f0d;
  box-shadow:inset 0 1px 2px rgba(0,0,0,.25), 0 0 0 3px var(--accent-soft);
}
textarea{resize:vertical;min-height:88px;font-family:var(--mono);font-size:12.5px;line-height:1.65}
select{
  cursor:pointer;appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='11' height='11' viewBox='0 0 24 24' fill='none' stroke='%236b7480' stroke-width='2.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 12px center;padding-right:32px;
}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.hint{font-size:12px;color:var(--text-3);line-height:1.55;letter-spacing:.005em}

/* ---------- 拖拽投放区 ---------- */
.drop{
  border:1.5px dashed var(--border-2); border-radius:var(--r-lg);
  padding:20px 14px; display:flex; flex-direction:column; align-items:center; gap:8px;
  cursor:pointer; transition:border-color 180ms var(--ease-out),
    background 180ms var(--ease-out), box-shadow 180ms var(--ease-out);
  background:#0a0d11; text-align:center;
}
.drop:hover{border-color:#3a4653;background:#0d1116}
.drop.hot{
  border-color:var(--accent); background:#0a1a15;
  box-shadow:0 0 0 4px var(--accent-soft);
}
.drop .ico{color:var(--text-3);transition:color 180ms var(--ease-out),transform 180ms var(--ease-out)}
.drop.hot .ico{color:var(--accent);transform:translateY(-3px)}
.drop .t1{font-size:13.5px;color:var(--text);font-weight:500;letter-spacing:-.005em}
.drop .t1 em{font-style:normal;color:var(--accent);font-weight:500}
.drop .t2{font-size:11.5px;color:var(--text-3);letter-spacing:.015em}
.up{display:none;flex-direction:column;gap:7px;width:100%;margin-top:4px}
.up .name{
  font-size:11.5px;color:var(--text-2);font-family:var(--mono);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.up .track{height:4px;background:#090c10;border-radius:2px;overflow:hidden}
.up .go{
  height:100%;width:0;border-radius:2px;
  background:linear-gradient(90deg,var(--accent-2),var(--blue));
  transition:width 140ms var(--ease-out);
}

/* ---------- 全屏投放遮罩 ---------- */
#veil{
  position:fixed;inset:0;z-index:900;display:none;
  background:rgba(6,9,12,.78);
  backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);
  align-items:center;justify-content:center;
}
#veil.on{display:flex}
#veil .box{
  border:2px dashed var(--accent); border-radius:var(--r-xl); padding:52px 76px;
  display:flex;flex-direction:column;align-items:center;gap:14px;
  background:rgba(10,26,21,.92); box-shadow:var(--sh-lg);
  animation:veilIn 220ms var(--ease-out);
}
@keyframes veilIn{
  from{opacity:0;transform:scale(.96)}   /* 不从 scale(0) 起——没有东西凭空出现 */
  to{opacity:1;transform:scale(1)}
}
#veil .box .big{font-size:32px;line-height:1}
#veil .box .t1{font-size:16px;font-weight:600;color:var(--accent);letter-spacing:-.01em}
#veil .box .t2{font-size:12.5px;color:var(--text-2)}

/* ---------- 按钮 ---------- */
.btn{
  border:1px solid var(--border-2); background:var(--surface-3); color:var(--text);
  padding:9px 15px; border-radius:var(--r-md); cursor:pointer;
  font-family:var(--font); font-size:13px; font-weight:500; line-height:1;
  transition:background 150ms var(--ease-out), border-color 150ms var(--ease-out),
             transform 150ms var(--ease-out);
  display:inline-flex; align-items:center; gap:7px; justify-content:center;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.04);
}
.btn:hover{background:var(--surface-4);border-color:#3a424d}
.btn:active{transform:scale(.97)}      /* 按下瞬间的反馈，emil 的核心手法 */
.btn.primary{
  background:linear-gradient(180deg,var(--accent),var(--accent-2));
  color:#04150e; border:none; font-weight:600;
  padding:13px; font-size:14.5px; letter-spacing:.005em;
  box-shadow:0 1px 0 rgba(255,255,255,.18) inset, 0 3px 10px rgba(61,220,151,.22);
}
.btn.primary:hover{filter:brightness(1.07);box-shadow:0 1px 0 rgba(255,255,255,.18) inset,0 4px 16px rgba(61,220,151,.32)}
.btn.primary:disabled{opacity:.4;cursor:not-allowed;filter:none;box-shadow:none;transform:none}
.btn.sm{padding:6px 12px;font-size:12.5px;border-radius:var(--r-sm)}
.btn.ghost{background:transparent;border-color:var(--border)}
.btn.ghost:hover{background:var(--surface-3);border-color:var(--border-2)}

/* ---------- 高级选项 ---------- */
.adv{border-top:1px solid var(--border);padding-top:14px}
.adv summary{
  cursor:pointer;font-size:12.5px;color:var(--text-2);font-weight:500;
  user-select:none;list-style:none;letter-spacing:.01em;
  transition:color 150ms var(--ease-out);
}
.adv summary:hover{color:var(--text)}
.adv summary::-webkit-details-marker{display:none}
.adv summary:before{content:"▸";display:inline-block;margin-right:7px;color:var(--text-3);
  transition:transform 180ms var(--ease-out)}
.adv[open] summary:before{transform:rotate(90deg)}
.adv .body{display:flex;flex-direction:column;gap:12px;padding-top:14px}

/* ---------- 右栏：进度 ---------- */
.progress-wrap{
  padding:15px 24px 12px; border-bottom:1px solid var(--border);
  background:var(--surface); flex:0 0 auto;
}
.ptop{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:9px}
.stage{font-size:12.5px;color:var(--text-2);letter-spacing:.005em}
.pct{
  font-size:13px;color:var(--accent);font-weight:600;
  font-variant-numeric:tabular-nums;letter-spacing:.02em;
}
.bar{
  height:5px;background:#090c10;border-radius:3px;overflow:hidden;
  border:1px solid #1a1f26;box-shadow:inset 0 1px 2px rgba(0,0,0,.4);
}
.fill{
  height:100%;width:0;border-radius:3px;
  background:linear-gradient(90deg,var(--accent-2),var(--accent),var(--blue));
  transition:width 380ms var(--ease-out);
}
.fill.idle{background:#1e242c}
.body-scroll{flex:1;overflow-y:auto;padding:18px 24px;min-height:0}

/* ---------- 空状态 ---------- */
.empty{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  height:100%;color:var(--text-3);gap:16px;text-align:center;padding:40px;
}
.empty .big{font-size:40px;opacity:.28;line-height:1}
.empty p{margin:0;font-size:13.5px;max-width:430px;line-height:1.75;color:var(--text-2)}
.empty .hint{color:var(--text-3);font-size:12.5px}

/* ---------- 日志 ---------- */
#log{
  font-family:var(--mono);font-size:12px;line-height:1.85;
  white-space:pre-wrap;word-break:break-all;color:#8e9aa8;margin:0;
  background:#080b0e;border:1px solid var(--border);border-radius:var(--r-lg);
  padding:14px 16px;display:none;max-height:40vh;overflow-y:auto;
  box-shadow:inset 0 1px 3px rgba(0,0,0,.35);
}
#log .ok{color:var(--accent)}
#log .err{color:var(--red)}
#log .warn{color:var(--amber)}
#log .dim{color:#4d5560}

/* ---------- 结果卡片 ---------- */
.card{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--r-lg);
  padding:18px 20px;margin-bottom:14px;box-shadow:var(--sh-sm);
  transition:border-color 180ms var(--ease-out), box-shadow 180ms var(--ease-out);
}
.card:hover{border-color:var(--border-2)}
.card.hi{
  border-color:rgba(61,220,151,.22);
  background:linear-gradient(180deg,#0d1614,var(--surface));
  box-shadow:0 1px 0 rgba(61,220,151,.08) inset, var(--sh-md);
}
.card h3{
  margin:0 0 5px;font-size:15.5px;font-weight:600;
  letter-spacing:-.012em;
  display:flex;align-items:center;gap:9px;flex-wrap:wrap;
}
.badge{
  font-size:11px;padding:3px 9px;border-radius:20px;font-weight:500;
  letter-spacing:.03em;flex:0 0 auto;
}
.badge.sub{background:var(--blue-soft);color:#8fc0ff}
.badge.asr{background:var(--accent-soft);color:var(--accent)}
.stats{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}
.stat{
  background:#090c10;border:1px solid var(--border);border-radius:var(--r-md);
  padding:8px 13px;min-width:80px;
}
.stat i{
  display:block;font-size:10.5px;color:var(--text-3);font-style:normal;
  margin-bottom:3px;letter-spacing:.05em;text-transform:uppercase;
}
.stat b{
  font-size:14.5px;font-weight:600;font-variant-numeric:tabular-nums;
  letter-spacing:-.01em;
}
.acts{display:flex;flex-wrap:wrap;gap:8px}
.preview{
  margin-top:14px;background:#080b0e;border:1px solid var(--border);
  border-radius:var(--r-md);padding:16px 18px;
  font-size:13.5px;line-height:1.9;color:#c2ccd6;
  max-height:46vh;overflow-y:auto;
  letter-spacing:.005em;
}
.preview .t{color:var(--accent);font-family:var(--mono);font-size:11.5px;opacity:.9}
/* 预览排版：标题 / 时间戳 / 段落（元信息不重复展示，卡片统计区已有） */
.pv-title{
  font-size:16px;font-weight:600;color:var(--text);letter-spacing:-.012em;
  margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid var(--border);
}
.pv-ts{margin:16px 0 5px}
.pv-ts:first-of-type{margin-top:4px}
.pv-para{
  color:#c2ccd6;font-size:13.5px;line-height:1.95;
  margin-bottom:2px;letter-spacing:.008em;
}

/* ---------- 滚动条 ---------- */
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#1e242c;border-radius:5px;border:2px solid var(--bg)}
::-webkit-scrollbar-thumb:hover{background:#2e3640}

.flash{animation:fl 520ms var(--ease-out)}
@keyframes fl{from{background:#0f2620}to{background:transparent}}

.err-box{
  background:#231315;border:1px solid #4a2428;color:#ff9b9b;
  padding:12px 15px;border-radius:var(--r-md);font-size:12.5px;margin-bottom:12px;
  line-height:1.6;
}
kbd{
  background:var(--surface-3);border:1px solid var(--border);border-bottom-width:2px;
  border-radius:5px;padding:1px 6px;font-size:11px;font-family:var(--mono);
  color:var(--text-2);
}

/* ---------- 无障碍：减少动态 ---------- */
@media (prefers-reduced-motion: reduce){
  *{animation-duration:.01ms !important;transition-duration:.01ms !important}
  .btn:active{transform:none}
}

/* ---------- 窄窗口：上下堆叠 ---------- */
@media (max-width: 900px){
  main{grid-template-columns:1fr;grid-template-rows:minmax(0,auto) 1fr}
  .splitter{display:none}
  aside{
    max-height:46vh;border-right:none;border-bottom:1px solid var(--border);
    padding:16px 16px 18px;gap:14px;
  }
  .drop{padding:14px 10px;gap:6px}
  .drop .t2{display:none}
  textarea{min-height:64px}
  header{padding:11px 16px}
}
</style>
</head>
<body>
<header>
  <div class="logo">
    <svg width="17" height="17" viewBox="0 0 64 64" fill="none">
      <path d="M22 15l27 17-27 17z" fill="#2ee6a8"/>
      <rect x="14" y="50" width="36" height="4" rx="2" fill="#4b9eff"/>
    </svg>
  </div>
  <h1>视频转文案 <span>v2t · 全程本地</span></h1>
  <div class="spacer"></div>
  <button class="bili-btn" id="biliBtn" title="B站登录：登录后可直取官方 AI 字幕，零识别误差、秒级完成">
    <span class="b-dot"></span><span id="biliTxt">B站未登录</span>
  </button>
  <span class="dot" id="gpuDot"></span><span class="hint" id="gpuTxt" style="margin-right:12px"></span>
  <span class="ver">v__VERSION__</span>
</header>

<main>
  <aside>
    <div class="drop" id="drop">
      <svg class="ico" width="26" height="26" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="7 9 12 4 17 9"/><line x1="12" y1="4" x2="12" y2="16"/>
      </svg>
      <div class="t1">拖视频 / 音频到这里，<em>或点击选择</em></div>
      <div class="t2">支持 mp4 / mkv / mov / mp3 / m4a / wav…</div>
      <div class="up" id="up">
        <div class="name" id="upname"></div>
        <div class="track"><div class="go" id="upgo"></div></div>
      </div>
    </div>
    <input type="file" id="file" multiple hidden accept="video/*,audio/*">

    <div class="group">
      <label class="lbl">或粘贴视频链接 / 文件路径 <b>每行一个</b></label>
      <textarea id="input" placeholder="https://www.bilibili.com/video/BVxxxxxxx&#10;D:\素材\某视频.mp4&#10;https://youtu.be/xxxx"></textarea>
      <div class="hint">支持 <b style="color:var(--accent)">B站 / YouTube</b> 等上千站点，也支持<b style="color:var(--accent)">抖音 / 快手 / 小红书</b>（浏览器抓取）。多行 = 批量，依次处理。</div>
    </div>

    <div class="group">
      <label class="lbl">识别引擎</label>
      <select id="engine" title="自动：中文用 Qwen3-ASR（更强），英文用 Whisper（更稳）">
        <option value="auto">自动（中文Qwen / 英文Whisper）</option>
        <option value="qwen">Qwen3-ASR-1.7B（中文最强）</option>
        <option value="whisper">Whisper large-v3（英文最强）</option>
      </select>
    </div>

    <div class="group">
      <label class="lbl">模型</label>
      <select id="model"></select>
    </div>

    <div class="group">
      <label class="lbl">语言</label>
      <select id="language">
        <option value="auto">自动识别（推荐）</option>
        <option value="zh" selected>中文（普通话）</option>
        <option value="yue">粤语</option>
        <option value="en">英文</option>
        <option value="ja">日文</option>
        <option value="ko">韩文</option>
        <option value="fr">法文</option>
        <option value="de">德文</option>
        <option value="es">西班牙文</option>
        <option value="pt">葡萄牙文</option>
        <option value="it">意大利文</option>
        <option value="ru">俄文</option>
        <option value="ar">阿拉伯文</option>
        <option value="th">泰文</option>
        <option value="vi">越南文</option>
        <option value="id">印尼文</option>
        <option value="ms">马来文</option>
        <option value="tr">土耳其文</option>
        <option value="hi">印地文</option>
        <option value="nl">荷兰文</option>
        <option value="pl">波兰文</option>
        <option value="sv">瑞典文</option>
        <option value="da">丹麦文</option>
        <option value="fi">芬兰文</option>
        <option value="cs">捷克文</option>
        <option value="el">希腊文</option>
        <option value="hu">匈牙利文</option>
        <option value="ro">罗马尼亚文</option>
        <option value="mk">马其顿文</option>
        <option value="fil">菲律宾文</option>
        <option value="fa">波斯文</option>
      </select>
      <div class="hint" id="langHint">Qwen3-ASR 支持 30 种语言 + 22 种中文方言；自动识别对中英混说也适用。</div>
    </div>

    <div class="group">
      <label class="lbl">输出目录</label>
      <input id="outdir" spellcheck="false">
    </div>

    <details class="adv">
      <summary>高级选项</summary>
      <div class="body">
        <div class="row">
          <div class="group">
            <label class="lbl">设备</label>
            <select id="device">
              <option value="cuda">GPU (CUDA)</option>
              <option value="cpu">CPU</option>
            </select>
          </div>
          <div class="group">
            <label class="lbl">精度</label>
            <select id="compute_type">
              <option value="float16">float16</option>
              <option value="int8_float16">int8_float16</option>
              <option value="int8">int8</option>
            </select>
          </div>
        </div>
        <div class="group">
          <label class="lbl">抓 B站 AI 字幕用的浏览器</label>
          <select id="cookies_from_browser">
            <option value="">不用（匿名）</option>
            <option value="edge">Edge</option>
            <option value="chrome">Chrome</option>
            <option value="firefox">Firefox</option>
          </select>
        </div>
        <label class="hint" style="display:flex;gap:8px;align-items:center;cursor:pointer">
          <input type="checkbox" id="no_subs" style="width:auto;margin:0">
          跳过平台字幕，强制本地 ASR
        </label>
        <label class="hint" style="display:flex;gap:8px;align-items:center;cursor:pointer">
          <input type="checkbox" id="vad" checked style="width:auto;margin:0">
          静音过滤（剔除纯音乐/空白段）
        </label>
        <label class="hint" style="display:flex;gap:8px;align-items:center;cursor:pointer"
               title="抖音/快手/小红书默认用浏览器抓取。若遇到滑块验证，勾选后会弹出可见浏览器窗口供你手动通过。">
          <input type="checkbox" id="headed" style="width:auto;margin:0">
          显示抓取用的浏览器窗口（遇到验证时勾选）
        </label>
      </div>
    </details>

    <button class="btn primary" id="go">开始提取</button>
    <div class="hint" id="est"></div>
  </aside>

  <div class="splitter" id="splitter" title="拖动调整左栏宽度（双击重置）"></div>

  <section class="out">
    <div class="progress-wrap">
      <div class="ptop">
        <span class="stage" id="stage">就绪</span>
        <span class="pct" id="pct">—</span>
      </div>
      <div class="bar"><div class="fill idle" id="fill"></div></div>
    </div>
    <div class="body-scroll" id="scroll">
      <div class="empty" id="empty">
        <div class="big">▶</div>
        <p>把视频链接粘到左边，点「开始提取」。<br>
        有平台字幕时几秒出稿；没有字幕就用本地 GPU 跑 Whisper。</p>
        <p class="hint">全程离线，音频不离开这台电脑。</p>
      </div>
      <div id="results"></div>
      <pre id="log"></pre>
    </div>
  </section>
</main>

<div id="veil">
  <div class="box">
    <div class="big">⬇</div>
    <div class="t1">松手即可上传</div>
    <div class="t2">文件会拷到 D 盘临时目录，转写完自动删除</div>
  </div>
</div>

<script>
const TOKEN = "__TOKEN__";
const $ = id => document.getElementById(id);
let running = false;

async function api(path, body){
  const opt = {headers:{"X-Token":TOKEN}};
  if(body !== undefined){
    opt.method = "POST";
    opt.headers["Content-Type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const r = await fetch(path, opt);
  if(!r.ok) throw new Error(await r.text() || ("HTTP " + r.status));
  return r.json();
}

const fmt = n => n >= 60 ? `${Math.floor(n/60)}分${Math.round(n%60)}秒` : `${Math.round(n)}秒`;
const hhmm = s => {
  s = Math.max(0, s|0);
  return String(Math.floor(s/3600)).padStart(2,'0')+':'+
         String(Math.floor(s%3600/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');
};

let META = { models: [], qwen_models: [] };

/* 模型下拉框跟着「识别引擎」走：
   - qwen    → 列出本地 Qwen3-ASR 模型
   - whisper → 列出本地 whisper 模型
   - auto    → 禁用并说明由语言自动决定，避免用户误以为选了就生效 */
function syncModelOptions(){
  const sel = $("model");
  const engine = $("engine").value;
  const prev = sel.value;
  let items = [], disabled = false, hint = "";

  if(engine === "qwen"){
    items = META.qwen_models || [];
    if(!items.length) hint = "未找到本地 Qwen3-ASR 模型";
  } else if(engine === "whisper"){
    items = META.models || [];
    if(!items.length) hint = "未找到本地 whisper 模型";
  } else {
    disabled = true;
    hint = "由语言自动决定";
  }

  if(items.length){
    sel.innerHTML = items.map(x =>
      `<option value="${x.name}">${x.name} · ${x.size_mb}MB</option>`).join("");
    // 尽量保留原选择；否则选默认项
    if(items.some(x => x.name === prev)) sel.value = prev;
    else if(engine === "whisper"){
      const pref = items.find(x => x.name === "large-v3");
      sel.value = pref ? pref.name : items[0].name;
    } else {
      sel.value = items[0].name;
    }
  } else {
    sel.innerHTML = `<option value="">${hint || "无可用模型"}</option>`;
  }
  sel.disabled = disabled || !items.length;
  sel.title = disabled ? "自动模式下按语言选择引擎，模型由系统决定" : "";
}

async function loadMeta(){
  const m = await api("/api/meta");
  META = { models: m.models || [], qwen_models: m.qwen_models || [] };
  syncModelOptions();

  $("outdir").value = m.outdir;
  $("device").value = m.gpu.available ? "cuda" : "cpu";
  if(!m.gpu.available){
    $("compute_type").value = "int8";
    $("gpuDot").style.background = "#ffb454";
    $("gpuDot").style.boxShadow = "0 0 8px #ffb454";
    $("gpuTxt").textContent = "无可用 GPU，将用 CPU（慢）";
  } else {
    $("gpuTxt").textContent = `GPU 就绪 · ${m.gpu.count} 卡`;
  }
  if(m.history && m.history.length) renderHistory(m.history);
}

function renderHistory(items){
  // 有历史时隐藏空状态，否则两块内容会叠在一起把页面撑出滚动条
  $("empty").style.display = "none";
  const box = $("results");
  box.innerHTML = `<div class="card"><h3 style="font-size:13px;color:var(--dim)">最近转写</h3>
    <div class="acts" style="margin-top:10px;flex-wrap:wrap">` +
    items.slice(0,10).map(h =>
      `<button class="btn sm ghost hist" data-md="${encodeURIComponent(h.md||'')}"
        title="${h.md||''}">${h.title||'未命名'} <span class="hint">${h.time||''}</span></button>`).join("") +
    `</div></div>`;
  box.querySelectorAll(".hist").forEach(b => b.onclick = () => {
    const p = decodeURIComponent(b.dataset.md);
    if(p) window.location.href = `/api/raw?path=${encodeURIComponent(p)}&token=${TOKEN}`;
  });
}

function appendLog(text, cls){
  const el = $("log");
  el.style.display = "block";
  const span = document.createElement("span");
  if(cls) span.className = cls;
  span.textContent = text + "\n";
  el.appendChild(span);
  // 只让日志框自己滚到底，不拖动外层页面——
  // 日志已移到结果卡片下方，若强行把整页滚到底，用户就看不到刚出的结果了。
  el.scrollTop = el.scrollHeight;
}

function setBar(pct, stage){
  $("fill").classList.remove("idle");
  $("fill").style.width = Math.min(100, Math.max(0, pct)) + "%";
  $("pct").textContent = Math.round(pct) + "%";
  if(stage) $("stage").textContent = stage;
}

async function start(){
  if(running) return;
  const raw = $("input").value.trim();
  if(!raw){ $("input").focus(); return; }
  const sources = raw.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  if(!sources.length) return;

  running = true;
  $("go").disabled = true;
  $("go").textContent = "处理中…";
  $("empty").style.display = "none";
  $("results").innerHTML = "";
  $("log").textContent = "";
  $("log").style.display = "block";
  setBar(0, "提交中");

  const payload = {
    sources,
    outdir: $("outdir").value.trim(),
    model: $("model").value,
    engine: $("engine").value,
    language: $("language").value,
    device: $("device").value,
    compute_type: $("compute_type").value,
    cookies_from_browser: $("cookies_from_browser").value,
    no_subs: $("no_subs").checked,
    vad: $("vad").checked,
    headed: $("headed").checked,
  };

  let jobId;
  try{
    const r = await api("/api/run", payload);
    jobId = r.id;
  }catch(e){
    appendLog("[err] 提交失败：" + e.message, "err");
    reset(); return;
  }

  const es = new EventSource(`/api/stream?id=${jobId}&token=${TOKEN}`);
  es.onmessage = ev => {
    const d = JSON.parse(ev.data);
    if(d.type === "start"){
      setBar(0, d.total > 1 ? `共 ${d.total} 个任务` : "开始");
      appendLog(`▶ 已提交 ${d.total} 个源`, "ok");
    }
    else if(d.type === "item"){
      appendLog(`\n──── [${d.index+1}/${d.total}] ${d.src} ────`, "dim");
    }
    else if(d.type === "stage"){
      setBar(d.pct, d.stage);
    }
    else if(d.type === "progress"){
      setBar(d.pct, d.msg);
    }
    else if(d.type === "log"){
      const m = d.msg || "";
      const cls = /✅|完成/.test(m) ? "ok" : /\[err\]|失败|Error/.test(m) ? "err" : /警告|\[cookies\]|retry/.test(m) ? "warn" : "";
      appendLog(m, cls);
    }
    else if(d.type === "item_done"){
      renderResult(d.result);
    }
    else if(d.type === "end"){
      es.close();
      setBar(100, d.status === "done" ? "完成" : "失败");
      if(d.status !== "done" && d.error) appendLog("\n[err] " + d.error, "err");
      reset();
    }
  };
  es.onerror = () => { es.close(); appendLog("[err] 连接中断", "err"); reset(); };
}

function reset(){
  running = false;
  $("go").disabled = false;
  $("go").textContent = "开始提取";
}

function renderResult(r){
  if(!r) return;
  const box = $("results");
  if(!r.ok){
    box.insertAdjacentHTML("beforeend",
      `<div class="err-box">✕ ${r.error || "失败"}${r.src ? `<br><span class="hint">${r.src}</span>` : ""}</div>`);
    return;
  }
  const sub = r.source === "subtitle";
  const card = document.createElement("div");
  card.className = "card hi flash";
  card.innerHTML = `
    <h3>${escapeHtml(r.title || "未命名")}
      <span class="badge ${sub ? 'sub' : 'asr'}">${sub ? '平台字幕' : '本地 ASR'}</span>
    </h3>
    <div class="hint" style="font-size:11.5px">${escapeHtml(r.md || "")}</div>
    <div class="stats">
      <div class="stat"><i>时长</i><b>${hhmm(r.duration)}</b></div>
      <div class="stat"><i>段落</i><b>${r.paragraphs}</b></div>
      <div class="stat"><i>字数</i><b>${r.chars}</b></div>
      <div class="stat"><i>耗时</i><b>${fmt(r.elapsed)}</b></div>
      ${r.asr ? `<div class="stat"><i>模型</i><b style="font-size:12px">${escapeHtml(r.asr.split(" / ")[0])}</b></div>` : ""}
    </div>
    <div class="acts">
      <button class="btn sm" data-act="preview" data-md="${encodeURIComponent(r.md)}">收起预览</button>
      <button class="btn sm" data-act="open" data-path="${encodeURIComponent(r.outdir||'')}">打开目录</button>
      <a class="btn sm" href="/api/file?id=${encodeURIComponent(r.md)}&token=${TOKEN}" download>下载 .md</a>
      <a class="btn sm" href="/api/file?id=${encodeURIComponent(r.txt)}&token=${TOKEN}" download>下载 .txt</a>
      <a class="btn sm" href="/api/file?id=${encodeURIComponent(r.srt)}&token=${TOKEN}" download>下载 .srt</a>
    </div>
    <div class="preview"></div>
  `;
  box.prepend(card);

  const pvEl = card.querySelector(".preview");

  // 结果卡片插到最前面，把视口拉回顶部——日志在下方，否则用户可能正停在日志区
  $("scroll").scrollTop = 0;

  // 提取成功即展开正文——用户要的就是文案本身，不该再多点一次
  (async () => {
    try{
      const resp = await fetch(`/api/raw?path=${encodeURIComponent(r.md)}&token=${TOKEN}`);
      pvEl.innerHTML = renderMd(await resp.text());
      pvEl.style.display = "block";
    }catch(e){
      pvEl.innerHTML = `<span class="hint">预览加载失败：${escapeHtml(e.message)}</span>`;
      pvEl.style.display = "block";
    }
  })();

  card.querySelectorAll("[data-act]").forEach(btn => btn.onclick = async () => {
    const act = btn.dataset.act;
    if(act === "open"){
      await api("/api/open", {path: decodeURIComponent(btn.dataset.path)});
      return;
    }
    // 切换展开/收起（内容已预加载，无需再请求）
    const open = pvEl.style.display === "block";
    pvEl.style.display = open ? "none" : "block";
    btn.textContent = open ? "展开预览" : "收起预览";
  });
}

/* 把 v2t 产出的 .md 渲染成可读正文。
   刻意跳过开头的元信息列表（来源/UP主/时长/文案来源/抓取时间/段落数）——
   卡片上方的统计区已经显示了同样的信息，预览里再列一遍是重复噪音。
   预览只留「标题 + 时间戳 + 段落」。
   所有文本先 escapeHtml，再拼标签，避免文案里的 < > 破坏结构。 */
function renderMd(raw){
  const lines = raw.split(/\r?\n/);
  const out = [];
  let inPara = false;
  let seenBody = false;   // 见过第一个时间戳后，元信息区就结束了

  const closePara = () => { if(inPara){ out.push("</div>"); inPara = false; } };

  for(let i = 0; i < lines.length; i++){
    const line = lines[i];

    if(/^#\s+/.test(line)){
      closePara();
      out.push(`<div class="pv-title">${escapeHtml(line.replace(/^#\s+/, ""))}</div>`);
      continue;
    }
    if(/^##\s+\[/.test(line)){
      closePara();
      seenBody = true;
      const t = line.match(/^##\s+\[([^\]]+)\]/);
      out.push(`<div class="pv-ts"><span class="t">${escapeHtml(t ? t[1] : "")}</span></div>`);
      continue;
    }
    if(/^---\s*$/.test(line)){
      closePara();
      continue;
    }
    // 元信息列表：正文开始前整段丢弃
    if(/^-\s+/.test(line) && !seenBody){
      continue;
    }
    if(line.trim() === ""){
      continue;
    }
    // 同一时间戳下的连续文本行合并成一个段落
    if(!inPara){ out.push('<div class="pv-para">'); inPara = true; }
    else out.push("<br>");
    out.push(escapeHtml(line));
  }
  closePara();
  return out.join("");
}

function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}

$("go").onclick = start;
$("engine").addEventListener("change", syncModelOptions);
$("input").addEventListener("keydown", e => {
  if(e.key === "Enter" && (e.ctrlKey || e.metaKey)){ e.preventDefault(); start(); }
});

/* ---------------- 左栏宽度：可拖动 + 记住 ---------------- */
const MIN_W = 340, MAX_W = 760;
function applyAside(w){
  w = Math.max(MIN_W, Math.min(MAX_W, w));
  document.documentElement.style.setProperty("--aside-w", w + "px");
  localStorage.setItem("v2t.asideW", w);
  return w;
}
(function initAside(){
  const saved = parseInt(localStorage.getItem("v2t.asideW") || "0", 10);
  // 窄窗口下由 CSS 的媒体查询接管，这里只在大屏生效
  if(saved && window.innerWidth > 900) applyAside(saved);
})();

(function bindSplitter(){
  const sp = $("splitter");
  let dragging = false;
  sp.addEventListener("mousedown", e => {
    // 双击重置：不能依赖 dblclick 事件——上面的 preventDefault 会把
    // 浏览器合成的 dblclick 链吃掉，实测双击无反应。用 detail===2 判定第二击。
    if(e.detail === 2){
      applyAside(460);
      return;
    }
    e.preventDefault();
    dragging = true;
    sp.classList.add("dragging");
    document.body.classList.add("resizing");
  });
  window.addEventListener("mousemove", e => {
    if(!dragging) return;
    const mainLeft = document.querySelector("main").getBoundingClientRect().left;
    applyAside(e.clientX - mainLeft);
  });
  window.addEventListener("mouseup", () => {
    if(!dragging) return;
    dragging = false;
    sp.classList.remove("dragging");
    document.body.classList.remove("resizing");
  });
})();

/* ---------------- 拖拽上传 ---------------- */
const VIDEO_EXT = /\.(mp4|mkv|mov|avi|flv|wmv|webm|m4v|ts|mpg|mpeg|3gp|mp3|m4a|wav|aac|flac|ogg|wma|opus)$/i;

function isMedia(f){
  return /^(video|audio)\//.test(f.type) || VIDEO_EXT.test(f.name);
}

let dragDepth = 0;
function showVeil(){ $("veil").classList.add("on"); }
function hideVeil(){ $("veil").classList.remove("on"); $("drop").classList.remove("hot"); }

window.addEventListener("dragenter", e => {
  // 只在真的拖文件时才亮遮罩，拖文字/链接不打扰
  if(!e.dataTransfer || ![...(e.dataTransfer.types||[])].includes("Files")) return;
  e.preventDefault();
  dragDepth++;
  showVeil();
  $("drop").classList.add("hot");
});
window.addEventListener("dragover", e => {
  if(e.dataTransfer && [...(e.dataTransfer.types||[])].includes("Files")){
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }
});
window.addEventListener("dragleave", e => {
  e.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if(dragDepth === 0) hideVeil();
});
window.addEventListener("drop", e => {
  if(!e.dataTransfer) return;
  e.preventDefault();
  dragDepth = 0;
  hideVeil();
  const files = [...e.dataTransfer.files].filter(isMedia);
  if(!files.length){
    if(e.dataTransfer.files.length){
      appendLog("[warn] 拖入的文件不是音视频格式，已忽略", "warn");
    }
    return;
  }
  handleFiles(files);
});

$("drop").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", e => {
  const files = [...e.target.files].filter(isMedia);
  if(files.length) handleFiles(files);
  e.target.value = "";
});

function fmtSize(n){
  if(n >= 1073741824) return (n/1073741824).toFixed(2) + " GB";
  if(n >= 1048576) return (n/1048576).toFixed(1) + " MB";
  return Math.max(1, Math.round(n/1024)) + " KB";
}

// 用 XHR 而不是 fetch：只有 XHR 能给上传进度
function uploadOne(file, onPct){
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/upload?name=${encodeURIComponent(file.name)}&token=${TOKEN}`);
    xhr.upload.onprogress = e => {
      if(e.lengthComputable) onPct(e.loaded / e.total);
    };
    xhr.onload = () => {
      if(xhr.status === 200){
        try{ resolve(JSON.parse(xhr.responseText)); }
        catch(err){ reject(new Error("返回解析失败")); }
      } else {
        reject(new Error(xhr.responseText || ("HTTP " + xhr.status)));
      }
    };
    xhr.onerror = () => reject(new Error("网络中断"));
    xhr.onabort = () => reject(new Error("已取消"));
    xhr.send(file);
  });
}

async function handleFiles(files){
  if(running){
    appendLog("[warn] 正在处理，等这轮跑完再拖", "warn");
    return;
  }
  $("empty").style.display = "none";
  $("log").style.display = "block";
  $("up").style.display = "flex";

  const paths = [];
  for(let i = 0; i < files.length; i++){
    const f = files[i];
    const tag = files.length > 1 ? ` (${i+1}/${files.length})` : "";
    $("upname").textContent = `上传中 ${f.name} · ${fmtSize(f.size)}${tag}`;
    $("upgo").style.width = "0%";
    try{
      const r = await uploadOne(f, p => { $("upgo").style.width = (p*100).toFixed(1) + "%"; });
      $("upgo").style.width = "100%";
      appendLog(`[upload] ${f.name} → 已就位 (${fmtSize(r.size)})`, "ok");
      paths.push(r.path);
    }catch(err){
      appendLog(`[err] 上传失败 ${f.name}：${err.message}`, "err");
    }
  }

  $("up").style.display = "none";
  if(!paths.length) return;

  // 路径回填到输入框，让用户看得见到底转的是哪个文件
  const cur = $("input").value.trim();
  $("input").value = (cur ? cur + "\n" : "") + paths.join("\n");
  start();
}

loadMeta().catch(e => appendLog("[err] 初始化失败：" + e.message, "err"));

// ---------------- B站登录 ----------------
// 登录后 B站视频可直取官方 AI 字幕：1 秒出稿、零识别误差（不登录只能跑 ASR，术语易错）。
let biliPollTimer = null;

function renderBili(st){
  const btn = $("biliBtn"), txt = $("biliTxt");
  if(!btn || !txt) return;
  if(st && st.login_running){
    btn.classList.remove("on");
    txt.textContent = "等待扫码…";
    btn.title = "浏览器已打开，请用 B站 App 扫码登录";
    return;
  }
  if(st && st.logged_in){
    btn.classList.add("on");
    const d = st.days_left;
    txt.textContent = d >= 1 ? `B站已登录 · ${Math.floor(d)}天` : "B站已登录";
    btn.title = `已登录：${st.username || ""}\n剩余 ${d} 天\n点击可登出`;
  }else{
    btn.classList.remove("on");
    txt.textContent = "B站未登录";
    btn.title = "点击登录 B站：可直取官方 AI 字幕，零识别误差、秒级完成";
  }
}

async function refreshBili(){
  try{
    const r = await fetch(`/api/bili/status?token=${encodeURIComponent(TOKEN)}`);
    const st = await r.json();
    renderBili(st);
    // 登录进行中 → 继续轮询
    if(st.login_running){
      clearTimeout(biliPollTimer);
      biliPollTimer = setTimeout(refreshBili, 2000);
    }
    return st;
  }catch(e){ return null; }
}

async function doBiliLogin(){
  const btn = $("biliBtn");
  btn.disabled = true;
  try{
    const r = await fetch(`/api/bili/login?token=${encodeURIComponent(TOKEN)}`,
                          {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"});
    const j = await r.json();
    if(j.already_running){
      appendLog("[bili] 登录流程已在进行中，请在弹出的浏览器里扫码", "warn");
    }else{
      appendLog("[bili] 已打开浏览器，请用 B站 App 扫码登录…", "info");
    }
    clearTimeout(biliPollTimer);
    biliPollTimer = setTimeout(refreshBili, 1500);
  }catch(e){
    appendLog("[bili] 启动登录失败：" + e.message, "err");
  }finally{
    setTimeout(()=>{ btn.disabled = false; }, 1500);
  }
}

async function doBiliLogout(){
  if(!confirm("确定退出 B站登录？\\n\\n退出后 B站视频将无法直取字幕，会退回本地 ASR（术语可能识别错误）。"))
    return;
  try{
    await fetch(`/api/bili/logout?token=${encodeURIComponent(TOKEN)}`,
                {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"});
    appendLog("[bili] 已登出", "info");
    refreshBili();
  }catch(e){ appendLog("[bili] 登出失败：" + e.message, "err"); }
}

$("biliBtn").addEventListener("click", async () => {
  const st = await refreshBili();
  if(st && st.logged_in) doBiliLogout();
  else doBiliLogin();
});

refreshBili();
</script>
</body>
</html>
"""


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "v2t-webui"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # 静音默认访问日志
        pass

    # ---------- 工具
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _err(self, code: int, msg: str) -> None:
        self._send(code, msg.encode("utf-8"), "text/plain; charset=utf-8")

    def _authed(self, q: dict) -> bool:
        tok = self.headers.get("X-Token") or (q.get("token") or [""])[0]
        return tok == TOKEN

    # ---------- GET
    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path

        if p in ("/", "/index.html"):
            html = HTML_PAGE.replace("__TOKEN__", TOKEN).replace("__VERSION__", VERSION)
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return

        if p == "/api/ping":
            self._json({"app": "v2t", "version": VERSION})
            return

        if not self._authed(q):
            self._err(403, "invalid token")
            return

        if p == "/api/meta":
            self._json({
                "models": list_models(),          # whisper 模型
                "qwen_models": list_qwen_models(),  # Qwen3-ASR 模型
                "gpu": gpu_info(),
                "outdir": default_outdir(),
                "history": load_history(),
                "version": VERSION,
            })
            return

        if p == "/api/bili/status":
            if bili_auth is None:
                self._json({"ok": False, "logged_in": False,
                            "reason": "bili_auth 模块缺失"})
                return
            try:
                st = bili_auth.status()
            except Exception as e:  # noqa: BLE001
                st = {"logged_in": False, "reason": str(e)}
            with BILI_LOGIN_LOCK:
                st["login_running"] = BILI_LOGIN_STATE["running"]
                st["login_message"] = BILI_LOGIN_STATE["message"]
            self._json({"ok": True, **st})
            return

        if p == "/api/stream":
            self._stream((q.get("id") or [""])[0])
            return

        if p == "/api/file":
            self._file((q.get("id") or [""])[0], download=True)
            return

        if p == "/api/raw":
            self._file((q.get("path") or [""])[0], download=False)
            return

        self._err(404, "not found")

    def _file(self, path: str, download: bool) -> None:
        try:
            f = Path(path).resolve()
        except Exception:
            self._err(400, "bad path")
            return
        if not f.is_file():
            self._err(404, "file not found")
            return
        # 只允许读 v2t 产出目录与用户指定输出目录下的文件，避免变成任意文件读取
        low = str(f).lower()
        if not any(low.startswith(r.lower()) for r in ALLOWED_ROOTS):
            self._err(403, "path not allowed")
            return
        data = f.read_bytes()
        ctype = "text/plain; charset=utf-8"
        extra = {}
        if download:
            from urllib.parse import quote
            extra["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(f.name)}"
        self._send(200, data, ctype, extra)

    def _upload(self, q: dict) -> None:
        """接收拖拽/选择上传的原始文件字节，流式落盘到 D 盘。

        为什么不用 multipart：标准库 Python 3.13+ 已移除 cgi，手写 multipart
        解析又容易踩边界 bug。直接 PUT 原始字节 + 文件名走 query，服务端只需
        按 Content-Length 拷贝，简单且不会整块进内存。
        """
        name = (q.get("name") or [""])[0]
        try:
            total = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._err(400, "bad Content-Length")
            return
        if total <= 0:
            self._err(400, "empty body")
            return

        # 只取纯文件名，挡掉 ../ 和盘符，防止写到上传目录之外
        safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", name).strip(" ._") or "upload.bin"
        safe = Path(safe).name[:120]

        # 每次上传一个独立子目录：既能保留原始文件名（产出标题才好看），
        # 又不会因同名文件互相覆盖。产出文件名 = 原文件名，没有时间戳前缀。
        sub = UPLOAD_DIR / str(int(time.time() * 1000))
        sub.mkdir(parents=True, exist_ok=True)
        dst = sub / safe

        got = 0
        try:
            with open(dst, "wb") as f:
                while got < total:
                    chunk = self.rfile.read(min(1 << 20, total - got))
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
        except Exception as e:  # noqa: BLE001
            try:
                dst.unlink(missing_ok=True)
            except Exception:
                pass
            self._err(500, f"写入失败：{e}")
            return

        if got != total:
            try:
                dst.unlink(missing_ok=True)
            except Exception:
                pass
            self._err(400, f"上传不完整：{got}/{total} 字节")
            return

        # 记进白名单，转写后 /api/file 才能读到（虽然它是源文件，不需要下载）
        self._json({"path": str(dst), "name": safe, "size": got})
    def _stream(self, job_id: str) -> None:
        job = JOBS.get(job_id)
        if job is None:
            self._err(404, "no such job")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        cursor = 0
        try:
            while True:
                with job.cond:
                    if len(job.events) <= cursor and job.status in ("queued", "running"):
                        job.cond.wait(0.8)
                    batch = job.events[cursor:]
                if batch:
                    for ev in batch:
                        self.wfile.write(b"data: " + json.dumps(ev, ensure_ascii=False).encode("utf-8") + b"\n\n")
                    cursor += len(batch)
                    self.wfile.flush()
                    if batch[-1].get("type") == "end":
                        break
                else:
                    # 心跳，防代理掐断
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    if job.status in ("done", "error"):
                        break
        except (BrokenPipeError, ConnectionResetError):
            pass
        try:
            self.close_connection = True
        except Exception:
            pass

    # ---------- POST
    def do_POST(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if not self._authed(q):
            self._err(403, "invalid token")
            return

        # 上传必须是原始字节流，且要在 JSON 解析之前分流——
        # 否则 json.loads(read(Content-Length)) 会把整个几 GB 的视频读进内存直接爆掉。
        if u.path == "/api/upload":
            self._upload(q)
            return

        n = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._err(400, "bad json")
            return

        if u.path == "/api/run":
            srcs = [s.strip() for s in (payload.get("sources") or []) if str(s).strip()]
            if not srcs:
                self._err(400, "empty sources")
                return
            outdir = payload.get("outdir") or default_outdir()
            cfg = load_config()
            cfg["outdir"] = outdir
            save_config(cfg)
            jid = secrets.token_hex(8)
            job = Job(jid, srcs, payload)
            with JOBS_LOCK:
                JOBS[jid] = job
                # 只留最近 40 个任务，防内存涨
                if len(JOBS) > 40:
                    for k in list(JOBS.keys())[: len(JOBS) - 40]:
                        JOBS.pop(k, None)
            WORK_QUEUE.put(job)
            self._json({"id": jid})
            return

        if u.path == "/api/bili/login":
            if bili_auth is None:
                self._err(500, "bili_auth 模块缺失")
                return
            with BILI_LOGIN_LOCK:
                if BILI_LOGIN_STATE["running"]:
                    self._json({"ok": True, "already_running": True})
                    return
                BILI_LOGIN_STATE.update(running=True, message="正在打开浏览器…",
                                        started=time.time())

            def _worker() -> None:
                try:
                    r = bili_auth.login()
                    msg = (f"登录成功，有效期 {r.get('days_left')} 天"
                           if r.get("ok") else f"失败：{r.get('reason')}")
                except Exception as e:  # noqa: BLE001
                    msg = f"异常：{type(e).__name__}: {e}"
                with BILI_LOGIN_LOCK:
                    BILI_LOGIN_STATE.update(running=False, message=msg)

            threading.Thread(target=_worker, daemon=True).start()
            self._json({"ok": True, "started": True})
            return

        if u.path == "/api/bili/logout":
            if bili_auth is None:
                self._err(500, "bili_auth 模块缺失")
                return
            try:
                r = bili_auth.logout()
                self._json({"ok": True, "removed": r.get("removed", [])})
            except Exception as e:  # noqa: BLE001
                self._err(500, str(e))
            return

        if u.path == "/api/open":
            path = payload.get("path") or ""
            try:
                if not path or not Path(path).exists():
                    self._err(404, f"目录不存在：{path}")
                    return
                os.startfile(path)  # noqa: S606
            except Exception as e:
                self._err(500, str(e))
                return
            self._json({"ok": True})
            return

        if u.path == "/api/shutdown":
            self._json({"ok": True})
            threading.Timer(0.3, lambda: os._exit(0)).start()
            return

        self._err(404, "not found")


# ------------------------------------------------------------------ 启动
def find_port(start: int) -> int:
    for p in range(start, start + 60):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"端口 {start}~{start+60} 全被占用")


def already_running(port: int) -> bool:
    """同一端口已经有 v2t 服务在跑，就不重复启动，直接开浏览器。"""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=0.6) as r:
            return b'"v2t"' in r.read()
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="v2t WebUI")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    # 清掉超过 24 小时的上传残留（上次转到一半崩了之类）
    try:
        if UPLOAD_DIR.is_dir():
            now = time.time()
            for f in UPLOAD_DIR.rglob("*"):
                try:
                    if f.is_file() and now - f.stat().st_mtime > 86400:
                        f.unlink(missing_ok=True)
                except Exception:
                    pass
            for d in sorted(UPLOAD_DIR.rglob("*"), reverse=True):
                if d.is_dir():
                    try:
                        d.rmdir()   # 只删空的
                    except OSError:
                        pass
    except Exception:
        pass

    if already_running(args.port):
        url = f"http://127.0.0.1:{args.port}/"
        print(f"v2t WebUI 已在运行：{url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    port = find_port(args.port)
    threading.Thread(target=worker, daemon=True).start()

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"

    print("=" * 52)
    print(f"  v2t WebUI v{VERSION}")
    print(f"  地址：{url}")
    print(f"  模型：{', '.join(m['name'] for m in list_models()) or '未找到'}")
    print(f"  GPU ：{'可用 ' + str(gpu_info().get('count')) + ' 卡' if gpu_info().get('available') else '不可用（将用 CPU）'}")
    print("=" * 52)
    print("  关闭方式：Ctrl+C，或直接关掉这个窗口")
    sys.stdout.flush()

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
