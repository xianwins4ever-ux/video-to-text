# -*- coding: utf-8 -*-
"""浏览器嗅探抓取通道 —— 专治 yt-dlp 打不动的平台。

为什么需要它
------------
实测 yt-dlp 对抖音已失效：
    ERROR: [Douyin] <id>: Fresh cookies (not necessarily logged in) are needed
且**带 cookies 也无解**——抖音对服务端请求返回验证码中间页。快手/小红书同理。

本模块换了一条路：用真浏览器打开分享页，让浏览器自己去算 a_bogus / X-Bogus
签名、自己过门禁；我们只在旁边「偷听」网络响应，抄下平台已经拿到的媒体直链，
再用普通 HTTP 下载。**不逆向任何签名算法**，所以平台改版时基本不受影响。

双策略，互为兜底
----------------
1. 拦 XHR JSON（如抖音 aweme/detail）—— 最准，还能顺带拿到标题/作者/时长
2. 拦媒体响应（video/mp4）—— 不依赖 API 路径，换平台也能活

用法
----
    python browser_fetch.py "<url>" --outdir D:\\out [--headed]

公开接口：
    needs_browser(url) -> bool      判断该链接是否必须走浏览器通道
    fetch(url, workdir, ...) -> dict 返回 {ok, title, author, media, ...}
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import sys
from pathlib import Path

TOOLS = Path(r"D:\tools\v2t")
FFMPEG_DIR = Path(r"D:\hyperframes-tools\ffmpeg")
FFMPEG = FFMPEG_DIR / "ffmpeg.exe"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0")

# 这些站 yt-dlp 支持差或已失效，必须走浏览器
BROWSER_ONLY_HOSTS = (
    "douyin.com", "iesdouyin.com",
    "kuaishou.com", "gifshow.com", "chenzhongtech.com",
    "xiaohongshu.com", "xhslink.com",
)

# 常见的媒体直链特征（用于兜底拦截）
MEDIA_HINTS = (".mp4", ".m4a", ".mp3", ".aac", "/play/", "media-video", "media-audio")


def needs_browser(url: str) -> bool:
    low = url.lower()
    return any(h in low for h in BROWSER_ONLY_HOSTS)


def _log(msg: str) -> None:
    print(f"[browser] {msg}", file=sys.stderr, flush=True)


def _pick_url_from_aweme(aweme: dict) -> str | None:
    """从抖音 aweme 对象里挑最高码率的直链。"""
    video = aweme.get("video") or {}
    best, best_rate = None, -1
    for band in (video.get("bit_rate") or []):
        urls = (band.get("play_addr") or {}).get("url_list") or []
        rate = band.get("bit_rate") or 0
        if urls and rate > best_rate:
            best, best_rate = urls[0], rate
    if best:
        return best
    for key in ("play_addr", "download_addr", "play_addr_h264"):
        urls = (video.get(key) or {}).get("url_list") or []
        if urls:
            return urls[0]
    return None


def _parse_meta(details: list[dict]) -> tuple[str, str, int, str | None]:
    """从抓到的 detail JSON 里取 标题/作者/时长/直链。"""
    for data in details:
        aweme = data.get("aweme_detail") or {}
        if not aweme and data.get("aweme_list"):
            aweme = data["aweme_list"][0]
        if not aweme:
            continue
        title = (aweme.get("desc") or "").strip()
        author = ((aweme.get("author") or {}).get("nickname") or "").strip()
        dur = int((aweme.get("video") or {}).get("duration") or 0)
        return title, author, dur, _pick_url_from_aweme(aweme)
    return "", "", 0, None


def _download(url: str, dst: Path, referer: str) -> tuple[bool, str]:
    """下载直链。抖音 CDN 校验 Referer，不带会被 403。"""
    try:
        import requests
    except ImportError:
        return False, "缺少 requests 库"

    headers = {
        "User-Agent": UA,
        "Referer": referer or "https://www.douyin.com/",
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Range": "bytes=0-",
    }
    try:
        with requests.get(url, headers=headers, stream=True, timeout=90) as r:
            if r.status_code not in (200, 206):
                return False, f"HTTP {r.status_code}"
            got = 0
            with open(dst, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
                        got += len(chunk)
            if got < 1024:
                return False, f"文件过小（{got} 字节），直链可能已失效"
            return True, f"{got / 1048576:.1f} MB"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def fetch(url: str, workdir: Path, *, headed: bool = False,
          timeout_ms: int = 45000, settle_ms: int = 7000,
          keep_video: bool = True, attempts: int = 2) -> dict:
    """打开页面、嗅探直链、下载。

    headed=True 会弹出可见浏览器窗口——遇到滑块验证时用户能手动过一下。
    attempts>1 时失败自动重试（抖音风控是间歇性的，重试常能过）。
    """
    last: dict = {}
    for i in range(1, max(1, attempts) + 1):
        res = _fetch_once(url, workdir, headed=headed, timeout_ms=timeout_ms,
                          settle_ms=settle_ms)
        if res.get("ok"):
            return res
        last = res
        if i < attempts:
            _log(f"第 {i} 次抓取失败（{res.get('error', '未知')}），"
                 f"{4 * i}s 后重试…")
            time.sleep(4 * i)
    return last


def _fetch_once(url: str, workdir: Path, *, headed: bool,
                timeout_ms: int, settle_ms: int) -> dict:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"ok": False, "error": "未安装 playwright（pip install playwright）"}

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    details: list[dict] = []
    media: list[tuple[str, int]] = []

    def on_response(resp) -> None:
        try:
            u = resp.url
            if "/aweme/detail/" in u and resp.status == 200:
                try:
                    details.append(resp.json())
                except Exception:
                    pass
                return
            if resp.status not in (200, 206):
                return
            ctype = (resp.headers.get("content-type") or "").lower()
            if "video/mp4" in ctype or any(h in u.lower() for h in MEDIA_HINTS):
                cl = resp.headers.get("content-length") or ""
                try:
                    size = int(cl)
                except ValueError:
                    m = re.search(r"/(\d+)$", resp.headers.get("content-range") or "")
                    size = int(m.group(1)) if m else 0
                media.append((u, size))
        except Exception:
            pass

    _log(f"打开 {url}")
    final_url = url
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=not headed)
            ctx = browser.new_context(user_agent=UA, locale="zh-CN",
                                      viewport={"width": 1440, "height": 900})
            page = ctx.new_page()
            page.on("response", on_response)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            except Exception as e:  # noqa: BLE001
                _log(f"页面加载告警（继续）：{type(e).__name__}")
            page.wait_for_timeout(settle_ms)
            final_url = page.url

            title, author, dur_ms, direct = _parse_meta(details)

            if not direct and not media:
                _log("首轮未命中，延长等待…")
                page.wait_for_timeout(6000)
                title, author, dur_ms, direct = _parse_meta(details)

            if not direct and media:
                media.sort(key=lambda x: x[1], reverse=True)
                direct = media[0][0]
                _log(f"从媒体响应兜底拿链（{media[0][1] / 1048576:.1f} MB）")

            if not direct:
                browser.close()
                return {"ok": False, "page_url": final_url,
                        "error": "没抓到媒体直链（可能私密/付费/需登录，或验证未过）"}

            dst = workdir / "browser_source.mp4"
            ok, info = _download(direct, dst, referer=final_url)
            browser.close()

            if not ok:
                return {"ok": False, "page_url": final_url, "error": f"下载失败：{info}"}
            _log(f"下载完成 {info}")

    except Exception as e:  # noqa: BLE001
        return {"ok": False, "page_url": final_url, "error": f"{type(e).__name__}: {e}"}

    return {
        "ok": True,
        "title": title or "video",
        "author": author,
        "duration_ms": dur_ms,
        "media": str(dst),
        "page_url": final_url,
    }


def extract_audio(video: Path, out_wav: Path) -> tuple[bool, str]:
    """抽音轨为 16k 单声道 wav，供 faster-whisper 使用。"""
    cmd = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", str(out_wav)]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0 or not out_wav.exists():
        return False, (r.stderr or "")[:300]
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(description="浏览器嗅探抓取（抖音/快手/小红书）")
    ap.add_argument("url")
    ap.add_argument("--outdir", default=str(TOOLS / "out" / "_browser"))
    ap.add_argument("--headed", action="store_true", help="弹出可见浏览器，便于手动过验证")
    a = ap.parse_args()

    wd = Path(a.outdir)
    res = fetch(a.url, wd, headed=a.headed)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
