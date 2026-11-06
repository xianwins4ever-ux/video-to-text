# video-to-text

> 视频链接 → 带时间戳的文字稿。**B站直接抓官方 AI 字幕，1 秒出稿，零识别误差。**
> 全程本地运行，音频不离开你的机器。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](#)

---

## 它解决什么问题

把视频变成可以搜索、引用、喂给 AI 的文本。典型用途：

- 看到一期好视频，想留下要点而不是二刷
- 在视频里找"他到底在哪一分钟讲的这个方法"
- 把视频内容整理进笔记库（Obsidian / Notion）
- 给 AI 助手提供视频里的信息（大多数 AI 读不了视频）

## 核心优势：不跑 ASR 也能拿到准确文案

市面上大部分方案都是「下载音频 → 跑 Whisper」。**这条路对中文专业术语极不友好**：

| | 本项目的 B站通道 | 通用 ASR 方案 |
|---|---|---|
| 8 分钟视频耗时 | **1.3 秒** | 100+ 秒 |
| `agent` | ✅ 正确 | ❌ 听成「A进」 |
| `[DSH]` | ✅ 正确 | ❌ 听成「Hynix」 |
| `Harness` | ✅ 正确 | ❌ 各种错拼 |

原因：B站用自家模型给几乎所有视频都生成了 AI 字幕（播放器字幕菜单里带 **(A)** 标记的那种），
这份字幕**现成、零误差**，只是接口要求登录才给。本项目带登录态直接下载，不做任何识别。

**没有平台字幕时**才退回本地 ASR（faster-whisper / Qwen3-ASR）。

---

## 安装

### 前置要求

| 项 | 要求 | 说明 |
|---|---|---|
| Python | 3.10+ | |
| ffmpeg | 必需 | Windows: `winget install Gyan.FFmpeg` |
| 磁盘 | 约 50MB（不装 ASR）/ 15GB（装 ASR） | |

### 步骤

```bash
git clone https://github.com/xianwins4ever-ux/video-to-text.git
cd video-to-text

# 主程序依赖（轻量，只有字幕抓取和 UI）
pip install yt-dlp opencc-python-reimplemented

# 可选：抖音/快手/小红书支持
pip install playwright
playwright install msedge
```

### ffmpeg 放哪里

程序会自动找，顺序是：

1. 环境变量 `V2T_FFMPEG_DIR`
2. 系统 PATH
3. `<项目目录>/ffmpeg/`
4. 常见安装位置

找不到就报错并提示怎么装。确认探测结果：

```bash
python bin/paths.py
```

### 可选：本地 ASR

只有「视频没有平台字幕」时才用到。不装的话，这类视频会跳过而不是报错。

**方案 A：faster-whisper（轻，约 3GB）**

```bash
pip install faster-whisper
python bin/fetch_models.py large-v3    # 下载模型
```

**方案 B：Qwen3-ASR（中文更强，约 9GB，需 GPU）**

```bash
python -m venv venv_qwen
venv_qwen\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126
venv_qwen\Scripts\pip install transformers accelerate soundfile numpy
```

程序会自动检测 `venv_qwen/` 是否存在，在则优先用 Qwen3-ASR。

---

## 快速开始

### 1. 登录 B站（强烈建议，一次管 180 天）

```bash
v2t login
```

弹出浏览器扫码即可。**这是拿到准确字幕的前提。**

有效期约 **180 天**（不是浏览器上显示的 7 天 —— 服务端认的是 `SESSDATA` 内嵌时间戳）。
到期重扫，服务端不会自动续期。

```bash
v2t status      # 查状态和剩余天数
v2t logout      # 清除登录
```

### 2. 转写

```bash
v2t "https://www.bilibili.com/video/BVxxxxxxx"
v2t "https://www.youtube.com/watch?v=xxxx"
v2t "D:\videos\local.mp4"
```

输出三个文件：

| 文件 | 用途 |
|---|---|
| `<标题>.md` | **推荐**：带 `## [00:12:34]` 时间戳，能回视频定位 |
| `<标题>.txt` | 纯文本，直接喂给搜索/摘要 |
| `<标题>.srt` | 标准字幕，能拖进剪辑软件 |

### 3. 图形界面

```bash
python bin/webui.py
# 打开 http://127.0.0.1:7788/
```

拖文件、粘链接、批量排队，右上角有 B站登录状态按钮。只监听 127.0.0.1，带 token 校验。

---

## 平台支持

**最后实测：2026-11-06。** 下表是真实跑出来的结论，不是"理论上支持"。

| 平台 | 搜索 | 转文案 | 说明 |
|---|---|---|---|
| **B站** | ✅ 免登录 | ✅ **1 秒** | 登录后直取官方 AI 字幕，本项目的核心场景 |
| **YouTube** | ✅ 免登录 | ⚠️ 看视频 | 有字幕的很快；没有的只能 ASR |
| **抖音** | ❌ | ✅ 浏览器嗅探 | 单条链接可用，批量易触发风控 |
| **快手** | ❌ | ✅ 浏览器嗅探 | 同上；**无字幕功能**，只能 ASR |
| **小红书** | ❌ | ✅ 浏览器嗅探 | 单条链接可用 |
| **微信视频号** | ❌ | ❌ | 加密 + 需 PC 微信，做不到 |

### 关于 YouTube 的实话

多数中文搬运视频**真的没有字幕** —— 不是权限问题，是上传者没传、
平台也没生成。实测 5 个中文 AI 频道视频全部返回 `has no automatic captions`。
这类只能 ASR，术语准确率比不上 B站通道。

另外偶发 `Sign in to confirm you're not a bot`（IP 级风控），
换 `player_client`、镜像站都无效，需要换网络环境。

### 关于抖音搜索的实话

关键词搜索是**平台封闭**，不是本项目没做。登录后接口仍返回
`search_nil_type: verify_check`，连 CDP 连接原生浏览器也过不了
（开源项目 MediaCrawler 自 2024-06 起同样失败）。

**单条视频链接转文案仍然可用。**

### 关于快手的实话（2026-11-06 实测）

两件事都验证过了：

**① 搜索被 captcha 拦。** `POST /rest/v/search/feed` 返回：
```json
{"data": {"result": 400002, "url": "https://captcha.zt.kuaishou.com/iframe/index.html?captchaSession=..."}}
```
同时 `profile/get` 返回 `{"result": 109}`（未登录）。需要登录 + 过人机验证，
与抖音同类风控。

**② 平台没有可提取的字幕轨。** 三项证据：
- `<video>` 元素 `textTracks.length === 0`，页面 0 个 `<track>` 标签
- 全程 **0 个字幕类网络请求**
- 页面数据（`window.INIT_STATE`）只有 `caption`（**视频标题**，不是字幕）和 `capSrc`，
  没有任何 subtitle / closedCaption 字段

快手的字幕是创作者用**快影 / 云剪**在发布前加上的，属于**硬字幕（烧进画面）**，
不是 B站那种可下载的软字幕轨。所以只能 ASR。

> 佐证：开源项目 [video-transcriber](https://github.com/quanji081/video-transcriber)
> 也支持快手，走的同样是 `yt-dlp + ffmpeg + faster-whisper`，而不是字幕提取。

**单条视频链接转文案可用**，实测 3.5–8MB 视频，45–70 秒出稿（含 ASR）。

---

## 工作原理

```
链接
 ├─ B站？ → 带 cookie 调字幕接口 → 有则直取（1 秒，零误差）★
 │                              └ 无则↓
 ├─ yt-dlp 支持？ → 探测平台字幕 → 有则下载
 │                            └ 无则↓
 ├─ 抖音/快手/小红书？ → 浏览器嗅探抓媒体流 → 抽音轨
 │                                    └ ↓
 └────────────────────────────→ 本地 ASR（faster-whisper / Qwen3-ASR）
```

---

## 常用参数

| 参数 | 说明 |
|---|---|
| `--outdir DIR` | 输出目录（默认 `./视频文案提取/`） |
| `--language zh` | 语言，`zh` / `en` / `auto` |
| `--engine auto` | 引擎：`auto` / `qwen` / `whisper` |
| `--no-subs` | 跳过字幕，强制 ASR |
| `--proxy URL` | 网络代理（YouTube 等需要） |
| `--no-vad` | 关闭静音过滤（纯音乐视频会用到） |
| `--keep-temp` | 保留中间音频便于重跑 |

完整列表：`v2t --help`

## 环境变量

| 变量 | 作用 | 默认 |
|---|---|---|
| `V2T_HOME` | 项目根目录 | 自动推导 |
| `V2T_FFMPEG_DIR` | ffmpeg 所在目录 | 自动查找 |
| `V2T_MODELS_DIR` | 模型目录 | `<项目>/models` |
| `V2T_COOKIE_DIR` | 登录凭证目录 | `~/.v2t/cookies` |

---

## 性能参考

测试机：i7-14650HX + RTX 4070 Laptop，3.5 分钟视频。

| 通道 | 耗时 | 速度 |
|---|---|---|
| **B站原生字幕** | **1.3 秒** | — |
| ASR large-v3 / CUDA | 18 秒 | 10.4x 实时 |
| ASR large-v3 / CPU | 205 秒 | 0.8x 实时 |

> **没有 GPU 时 large-v3 比视频还慢**，实际不可用。这也是 B站通道价值高的原因。

---

## 目录结构

```
video-to-text/
├── v2t.cmd / v2t-webui.cmd     Windows 入口
├── bin/
│   ├── v2t.py                  主程序（CLI + 核心流程）
│   ├── webui.py                浏览器界面
│   ├── paths.py                ★ 路径与环境探测（唯一真相源）
│   ├── bili_auth.py            ★ B站登录 / 登出 / 有效期
│   ├── bili_subs.py            ★ B站原生字幕直取
│   ├── browser_fetch.py        浏览器嗅探（抖音等）
│   ├── qwen_engine.py          Qwen3-ASR 适配器
│   └── fetch_*.py              模型 / CUDA 依赖下载
├── .github/                    issue / PR 模板
├── LICENSE                     MIT
└── README.md
```

---

## 已知限制

- **ASR 必然会错字**，中文人名、专有名词、方言尤其。要精确引用请回视频核对。
  这也是为什么 B站强烈建议登录 —— 直取官方字幕能完全避免这类错误。
- **纯音乐段落会被剔除**。这是有意的，避免幻听出歌词。要保留加 `--no-vad`。
- **微信视频号不支持**，技术上做不到。

## 故障排查

| 现象 | 处理 |
|---|---|
| B站走了 ASR、术语离谱 | 未登录。`v2t login` |
| `cookie 已失效` | 180 天到期，重新 `v2t login` |
| 找不到 ffmpeg | `python bin/paths.py` 看探测结果，装 ffmpeg 或设 `V2T_FFMPEG_DIR` |
| YouTube 报 not a bot | IP 风控，换网络/代理 |
| 转写 0 段且极快 | 静音检测误杀，加 `--no-vad` |

更多问题请[提 issue](https://github.com/xianwins4ever-ux/video-to-text/issues)。

---

## 贡献

欢迎 PR。动手前请读 [CONTRIBUTING.md](CONTRIBUTING.md)，有几条硬性约定
（不要硬编码路径、不要提交模型和 cookie、B站改动必须测登录/未登录两种状态）。

## 免责声明

本项目仅用于个人学习与研究，请勿用于大规模爬取或商业用途。
使用本工具产生的内容版权归原作者所有，请尊重创作者权益。
因使用不当产生的法律责任由使用者自行承担。

## 许可证

[MIT](LICENSE) © 2026 xianwins4ever-ux
