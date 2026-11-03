# -*- coding: utf-8 -*-
"""生成 v2t 图标（多尺寸 .ico）。

构图：深色圆角方块 + 青绿播放三角 + 渐弱文字线（代表「视频 → 文案」）。

关键：ICO 里每个尺寸单独绘制，而不是把一张 512 缩下去。
实测把大图缩到 16px 时，三条文字线会糊成一条灰带；所以小尺寸用
「简化版」（更大的三角 + 两条线），中大战用完整版。
"""
import io
import struct
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

OUT = paths.ASSETS
OUT.mkdir(parents=True, exist_ok=True)

BG = (13, 17, 23, 255)
BORDER = (34, 46, 59, 255)
GREEN = (46, 230, 168, 255)
BLUE = (75, 158, 255, 255)
GREY1 = (110, 135, 165, 255)
GREY2 = (78, 95, 115, 255)


def render(size: int, simple: bool) -> Image.Image:
    """按目标尺寸直接绘制，超采样 4x 后缩小做抗锯齿。"""
    k = 4
    S = size * k
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    pad = int(S * 0.035)
    radius = int(S * 0.21)
    d.rounded_rectangle((pad, pad, S - pad, S - pad), radius=radius, fill=BG)
    if not simple:
        d.rounded_rectangle((pad, pad, S - pad, S - pad), radius=radius,
                            outline=BORDER, width=max(1, int(S * 0.007)))

    if simple:
        # 小尺寸：三角放大居中，只留一条底线，保证缩小后可辨
        cx, cy = S * 0.44, S * 0.40
        tw, th = S * 0.28, S * 0.34
        d.polygon([(cx, cy - th / 2), (cx + tw, cy), (cx, cy + th / 2)], fill=GREEN)
        y0 = S * 0.74
        d.rounded_rectangle((S * 0.20, y0, S * 0.80, y0 + S * 0.075),
                            radius=S * 0.038, fill=BLUE)
    else:
        # 大尺寸：播放三角 + 三条渐弱文字线
        cx, cy = S * 0.40, S * 0.395
        tw, th = S * 0.225, S * 0.285
        d.polygon([(cx, cy - th / 2), (cx + tw, cy), (cx, cy + th / 2)], fill=GREEN)

        x0 = S * 0.185
        lines = [(0.605, 0.53, BLUE), (0.710, 0.41, GREY1), (0.815, 0.28, GREY2)]
        for y, w, col in lines:
            y0 = S * y
            d.rounded_rectangle((x0, y0, x0 + S * w, y0 + S * 0.055),
                                radius=S * 0.027, fill=col)

    return img.resize((size, size), Image.LANCZOS)


def build_ico(images: dict[int, Image.Image]) -> bytes:
    """手工拼 ICO：每个尺寸塞自己的 PNG，避免 PIL 自动缩放丢细节。"""
    header = struct.pack("<HHH", 0, 1, len(images))
    entries, blobs = b"", b""
    offset = 6 + 16 * len(images)
    for s in sorted(images):
        buf = io.BytesIO()
        images[s].save(buf, format="PNG", optimize=True)
        png = buf.getvalue()
        w = s if s < 256 else 0
        entries += struct.pack("<BBBBHHII", w, w, 0, 0, 1, 32, len(png), offset)
        offset += len(png)
        blobs += png
    return header + entries + blobs


sizes = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
images = {s: render(s, simple=(s <= 32)) for s in sizes}

(OUT / "icon.ico").write_bytes(build_ico(images))
images[256].save(OUT / "icon.png")

# 检查图：真实像素大小平铺，肉眼确认每个尺寸是否可辨
gap = 14
width = sum(sizes) + gap * (len(sizes) + 1)
sheet = Image.new("RGBA", (width, 256 + 2 * gap), (24, 28, 34, 255))
x = gap
for s in sizes:
    sheet.alpha_composite(images[s], (x, gap + (256 - s)))
    x += s + gap
sheet.save(OUT / "preview.png")

print(f"icon.ico {(OUT / 'icon.ico').stat().st_size} bytes · {len(sizes)} 个尺寸")
