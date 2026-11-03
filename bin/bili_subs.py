# -*- coding: utf-8 -*-
"""B站原生字幕直取 —— 优先于 yt-dlp / ASR。

为什么需要这个模块
------------------
yt-dlp 调 B站 `x/player/wbi/v2` 时**不带登录态**，接口返回 `subtitles: []` +
`need_login_subtitle: True`。于是 v2t 判定"该视频无字幕"，转去跑本地 ASR。
而实际上 B站早就用自家模型生成好了 AI 字幕（菜单里的"中文 (A)"，A = AI），
只是要求登录才能取。

obsidian-clipper-cn 的解法（我们复用其思路）：
    fetch(url, { credentials: "include" })   # 带着页面登录 cookie 请求
没有逆向，就是带凭证。

本模块：读本地保存的 B站 cookie → 调同一接口 → 下载字幕 JSON。
零 ASR、零识别误差、秒级完成。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
import http.cookiejar
from pathlib import Path

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0")

# cookie 由 bili_auth 统一管理（登录/登出/有效期），这里只读
try:
    import bili_auth
except ImportError:  # 单独运行本文件时
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bili_auth

MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52,
]


# 兼容早期版本：vsearch 曾把 cookie 存在 ~/.video-research/cookies
LEGACY_DIRS = (Path.home() / ".video-research" / "cookies",)


def cookie_file() -> Path | None:
    """返回可用的 cookie 文件路径（新位置优先，其次旧位置）。"""
    p = bili_auth.cookie_file()
    if p.exists():
        return p
    for d in LEGACY_DIRS:
        q = d / "bili.json"
        if q.exists():
            return q
    return None


def _load_cookies() -> http.cookiejar.CookieJar | None:
    p = cookie_file()
    if p is None:
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    cj = http.cookiejar.CookieJar()
    for c in raw:
        cj.set_cookie(http.cookiejar.Cookie(
            version=0, name=c.get("name", ""), value=c.get("value", ""),
            port=None, port_specified=False,
            domain=c.get("domain", ""), domain_specified=bool(c.get("domain")),
            domain_initial_dot=str(c.get("domain", "")).startswith("."),
            path=c.get("path", "/"), path_specified=True,
            secure=bool(c.get("secure")), expires=int(c.get("expires") or 0) or None,
            discard=False, comment=None, comment_url=None, rest={}))
    return cj


def _opener():
    cj = _load_cookies()
    if cj is None:
        return None, False
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", UA), ("Referer", "https://www.bilibili.com/")]
    return op, True


def _json(op, url: str):
    with op.open(url, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def _text(op, url: str) -> str:
    with op.open(url, timeout=20) as r:
        return r.read().decode("utf-8", errors="replace")


def is_bilibili(url: str) -> bool:
    return "bilibili.com" in url or bool(re.search(r"\bBV[0-9A-Za-z]{10}\b", url))


def extract_bvid(url: str) -> str | None:
    m = re.search(r"(BV[0-9A-Za-z]{10})", url)
    return m.group(1) if m else None


def fetch_bilibili_subtitle(url: str, prefer_lang: str = "zh") -> dict:
    """取 B站原生字幕。

    返回 {ok, reason, bvid, title, cid, duration, lan, lan_doc, cues:[{from,to,content}]}
    ok=False 时 reason 说明原因（未登录 / 视频真没字幕 / 网络错误）。
    """
    bvid = extract_bvid(url)
    if not bvid:
        return {"ok": False, "reason": "无法从链接解析出 BV 号"}

    op, logged_in = _opener()
    if op is None:
        return {"ok": False, "reason": "没有 B站 cookie，先运行 `v2t login`"}

    # 取 wbi key
    try:
        op.open("https://www.bilibili.com/", timeout=20).read()
        nav = _json(op, "https://api.bilibili.com/x/web-interface/nav")
    except Exception as e:
        return {"ok": False, "reason": f"网络错误: {e}"}

    is_login = bool((nav.get("data") or {}).get("isLogin"))
    if not is_login:
        return {"ok": False, "reason": "cookie 已失效，需重新 `vsearch login bili`"}

    wbi = (nav.get("data") or {}).get("wbi_img") or {}
    img = (wbi.get("img_url") or "").rsplit("/", 1)[-1].split(".")[0]
    sub = (wbi.get("sub_url") or "").rsplit("/", 1)[-1].split(".")[0]
    if not img or not sub:
        return {"ok": False, "reason": "拿不到 wbi key"}
    mixin = "".join((img + sub)[i] for i in MIXIN_KEY_ENC_TAB)[:32]

    def signed(path: str, params: dict) -> str:
        p = dict(params, wts=str(round(time.time())))
        p = dict(sorted(p.items()))
        p = {k: "".join(ch for ch in str(v) if ch not in "!'()*") for k, v in p.items()}
        q = urllib.parse.urlencode(p)
        p["w_rid"] = __import__("hashlib").md5((q + mixin).encode()).hexdigest()
        return f"https://api.bilibili.com{path}?" + urllib.parse.urlencode(p)

    # 视频信息
    try:
        info = _json(op, signed("/x/web-interface/wbi/view", {"bvid": bvid}))["data"]
    except Exception as e:
        return {"ok": False, "reason": f"取视频信息失败: {e}"}

    cid = info.get("cid")
    aid = info.get("aid")
    meta = {"bvid": bvid, "title": info.get("title"),
            "cid": cid, "duration": info.get("duration"),
            "author": (info.get("owner") or {}).get("name")}

    # 分P 支持
    p = None
    m = re.search(r"[?&]p=(\d+)", url)
    if m:
        pno = int(m.group(1))
        pages = info.get("pages") or []
        if 0 < pno <= len(pages):
            cid = pages[pno - 1].get("cid")
            meta["cid"] = cid

    # 取字幕轨（wbi 优先，失败回落老接口）
    tracks = []
    for endpoint, params in (
        ("/x/player/wbi/v2", {"bvid": bvid, "cid": cid, "aid": aid}),
        ("/x/player/v2", {"cid": cid, "bvid": bvid, "aid": aid}),
    ):
        try:
            r = _json(op, signed(endpoint, params) if "wbi" in endpoint
                      else f"https://api.bilibili.com{endpoint}?"
                      + urllib.parse.urlencode({k: v for k, v in params.items() if v}))
            tracks = ((r.get("data") or {}).get("subtitle") or {}).get("subtitles") or []
            if tracks:
                break
        except Exception:
            continue

    if not tracks:
        return {"ok": False, "reason": "该视频确实没有字幕（B站未生成 AI 字幕）", **meta}

    # 选中文轨
    def score(t):
        lang = f"{t.get('lan','')} {t.get('lan_doc','')}".lower()
        if "zh-cn" in lang or "中文" in lang: return 5
        if "zh-hans" in lang or "简体" in lang: return 4
        if "zh" in lang: return 3
        if "en" in lang: return 2
        return 1

    track = sorted(tracks, key=score, reverse=True)[0]
    surl = track.get("subtitle_url") or ""
    if surl.startswith("//"):
        surl = "https:" + surl
    if not surl:
        return {"ok": False, "reason": "字幕轨没有 URL", **meta}

    try:
        body = json.loads(_text(op, surl))
    except Exception as e:
        return {"ok": False, "reason": f"下载字幕失败: {e}", **meta}

    cues = [
        {"from": ln.get("from"), "to": ln.get("to"),
         "content": re.sub(r"\s+", " ", (ln.get("content") or "")).strip()}
        for ln in (body.get("body") or [])
        if isinstance(ln.get("from"), (int, float)) and (ln.get("content") or "").strip()
    ]
    if not cues:
        return {"ok": False, "reason": "字幕文件为空", **meta}

    return {"ok": True, "lan": track.get("lan"), "lan_doc": track.get("lan_doc"),
            "cues": cues, "source": "bilibili_native", **meta}


def to_markdown(result: dict) -> str:
    """转成带时间戳的 md。"""
    def ts(sec: float) -> str:
        s = int(sec)
        h, m, ss = s // 3600, (s % 3600) // 60, s % 60
        return f"{h:02d}:{m:02d}:{ss:02d}" if h else f"{m:02d}:{ss:02d}"

    lines = [
        f"# {result.get('title', '')}",
        "",
        f"- 来源：https://www.bilibili.com/video/{result.get('bvid')}",
        f"- UP主：{result.get('author', '')}",
        f"- 字幕：B站原生 AI 字幕（{result.get('lan_doc', '')} / {result.get('lan', '')}）",
        "- 抓取方式：B站接口直取（非 ASR，无识别误差）",
        f"- 句数：{len(result.get('cues', []))}",
        "",
        "---",
        "",
    ]
    for c in result.get("cues", []):
        lines.append(f"**[{ts(c['from'])}]** {c['content']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.bilibili.com/video/BV1fxni6XEwD/"
    r = fetch_bilibili_subtitle(url)
    if r["ok"]:
        print(f"✓ {r['title']}  ({r['lan_doc']})  {len(r['cues'])} 句")
        print(to_markdown(r)[:800])
    else:
        print(f"✗ {r['reason']}")
