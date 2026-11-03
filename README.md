# v2t —— 视频链接 → 文案

丢一个视频链接，拿到可检索的文字稿（带时间戳）。**全程本地，内容不出这台机器。**

## ⭐ 先做这一步：B站登录（强烈建议）

```powershell
D:\tools\v2t\v2t.cmd login
```

弹浏览器扫码，登录一次即可。

### 为什么要登录

B站用自家模型给**几乎所有视频**都生成了 AI 字幕（播放器字幕菜单里带 **(A)** 标记的那种）。
这份字幕是现成的、**零识别误差**的，下载只要 **1 秒**——但接口**要求登录**才能取：

| 状态 | 接口返回 | 结果 |
|---|---|---|
| 未登录 | `subtitles: []` + `need_login_subtitle: True` | 只能下音频跑 ASR，术语必错 |
| 已登录 | `subtitles: [{lan: "ai-zh", ...}]` | 直接下载，1 秒出稿 |

不登录的代价（实测）：

| | 登录后 | 未登录 |
|---|---|---|
| 耗时（8 分钟视频） | **1.3 秒** | 100+ 秒 |
| `agent` | ✅ 正确 | ❌ 听成「A进」 |
| `[DSH]` | ✅ 正确 | ❌ 听成「Hynix / Hunnes」 |
| `Harness` | ✅ 正确 | ❌ 各种错拼 |

> 原因：B站的 ASR 有平台热词库，你本地的 Qwen3-ASR 没有。

### 有效期

**约 180 天**，到期需重新扫码。

> 浏览器里看到的「登录有效期 1 天 / 7 天」是 cookie 的保存期限，
> B站服务端真正认的是 `SESSDATA` 里内嵌的时间戳（约 180 天）。
> v2t 会读取这个内嵌时间戳，所以不会像浏览器那样 7 天就失效。

服务端**不会**自动续期，所以 180 天后必须重扫。

### 命令

```powershell
D:\tools\v2t\v2t.cmd login     # 扫码登录
D:\tools\v2t\v2t.cmd status    # 查登录状态与剩余天数
D:\tools\v2t\v2t.cmd logout    # 清除本地登录信息
```

登录信息保存在 `~/.v2t/cookies/`，只在本机，不会上传。

### 登出后会怎样

B站视频自动退回本地 ASR（耗时变长、术语可能出错），其他站点不受影响。
v2t 会在检测到未登录时主动提示你。

---

## 两种用法

### A. 图形界面（推荐日常用）

双击桌面 **「视频转文案」** 图标 → 浏览器自动打开 `http://127.0.0.1:7788/`

- 粘链接 → 点「开始提取」，进度条和日志实时刷新
- 结果卡片直接预览文案、打开目录、下载 .md/.txt/.srt
- 多行输入 = 批量排队，一次交一批
- 已转过的记录留在「最近转写」里，点了就能看

> 重复双击不会开第二个实例，只会把已有窗口调出来（GPU 只有 8GB，并行会 OOM）。

手动启动：

```powershell
D:\tools\v2t\v2t-webui.cmd          # 无窗口
D:\tools\v2t\v2t-webui-debug.cmd    # 保留控制台，便于看报错
```

日志：`<v2t目录>\data\webui.log`
重新生成图标 / 快捷方式：`python bin\make_icon.py`、`python bin\create_shortcut.py`

### B. 命令行

```powershell
# 最简：丢链接
D:\tools\v2t\v2t.cmd "https://www.bilibili.com/video/BV1xxxxxxxxx"

# 本地视频文件
D:\tools\v2t\v2t.cmd "D:\some\video.mp4"

# 指定输出目录
D:\tools\v2t\v2t.cmd "<url>" --outdir "D:\素材\某项目\文案"
```

输出三个文件到 `--outdir`（默认 `当前目录\视频文案提取\`）：

| 文件 | 用途 |
|---|---|
| `<标题>.md` | **推荐**：带 `## [00:12:34]` 时间戳的段落稿，方便回视频里定位 |
| `<标题>.txt` | 纯文本，喂给 AI 提问/检索用 |
| `<标题>.srt` | 标准字幕，可导入剪辑软件 |

## 工作原理（三条通道，自动选路）

```
链接进来
  │
  ├─ 抖音 / 快手 / 小红书 ──► ① 浏览器嗅探（真 Edge 代算签名，抄直链）
  │                              yt-dlp 对这三家已失效，不再浪费时间
  │
  └─ B站 / YouTube / 其他 ──► ② 先探平台字幕 ──有──► 直接导出（秒级、零错字）
  │                                            └─无──► ③ yt-dlp 下音频 → 本地 ASR
  └─ 本地文件 ─────────────► 直接 ASR
```

**为什么抖音要单独一条通道**：实测 yt-dlp 报
`Fresh cookies (not necessarily logged in) are needed`，**带 cookies 也无解**——
抖音对服务端请求返回验证码中间页。浏览器嗅探让真浏览器自己算 `a_bogus` 签名、
自己过门禁，我们只偷听它已经拿到的直链，所以平台改版也不怕。

## 识别引擎（双引擎，自动选路）

| 引擎 | 中文 | 英文 | 说明 |
|---|---|---|---|
| **Qwen3-ASR-1.7B** | ✅ 强 | 一般 | 2025 年新模型，中文 CER 约是 Whisper 的一半 |
| **Whisper large-v3** | 一般 | ✅ 强 | 2023 年模型，干净英文朗读仍略胜 |

默认 `--engine auto`：**英文走 Whisper，其余走 Qwen**。

实测同一段抖音音频：

| Whisper large-v3 | Qwen3-ASR-1.7B |
|---|---|
| 视觉**动效** | 视觉**特效** |
| **A进**的完成最后一块**评论** | **AI** 完成最后一块**拼图** |
| 老司机更是**汉流加倍** | 老司机更是**汗流浃背** |

行业数据：中文 WenetSpeech **4.97% vs 9.86%**；极端噪声 **16.17% vs 63.17%**。

```powershell
D:\tools\v2t\v2t.cmd "<url>"                      # 自动选路（推荐）
D:\tools\v2t\v2t.cmd "<url>" --engine qwen        # 强制 Qwen（中文素材）
D:\tools\v2t\v2t.cmd "<url>" --engine whisper     # 强制 Whisper（英文素材）
```

## 常用参数

| 参数 | 说明 |
|---|---|
| `--model large-v3` | Whisper 模型（`--engine whisper` 时生效） |
| `--engine auto` | 识别引擎：`auto`(按语言选，默认) / `qwen` / `whisper` |
| `--device cuda` | 默认。GPU 加速，比 CPU 快 10~20 倍 |
| `--device cpu --compute-type int8` | 没显卡时用，慢 |
| `--language zh` | 语言，默认中文。英文视频用 `--language en` 或 `auto` |
| `--sub-lang zh-Hans` | 优先抓哪种字幕 |
| `--cookies-from-browser edge` | 借用浏览器 cookie（B站建议直接用 `v2t login`，更省事） |
| `--headed` | 浏览器抓取时弹出可见窗口（遇到滑块验证可手动过） |
| `--no-subs` | 跳过字幕，强制走 ASR |
| `--keep-temp` | 保留中间音频，便于换模型重跑 |
| `--retries 4` | 网络重试次数 |

## 环境变量（可选）

| 变量 | 作用 | 默认 |
|---|---|---|
| `V2T_HOME` | 工具根目录 | 自动按脚本位置推导 |
| `V2T_FFMPEG_DIR` | ffmpeg 所在目录 | 自动从 PATH / 常见位置查找 |
| `V2T_COOKIE_DIR` | 登录 cookie 存放目录 | `~/.v2t/cookies` |

> 默认值都是自适应的，把整个目录解压到任何位置都能直接跑，不需要改配置。

## 本机实测数据（i7-14650HX + RTX 4070 Laptop）

| 路径 | 3.5 分钟视频耗时 | 速度 |
|---|---|---|
| 平台字幕 | ~7 秒 | — |
| ASR small / CPU | 41 秒 | 5.2x 实时 |
| ASR medium / CPU | 158 秒 | 1.1x 实时 |
| ASR large-v3 / CPU | 205 秒 | 0.8x 实时 |
| ASR large-v3 / **CUDA** | **18 秒**（含首帧编译） | **10.4x 实时** |

> **结论**：没有 CUDA 时 large-v3 比视频时长还慢，实际不可用。GPU 加速是刚需。
> CUDA 实测数据（2026-11-01，70 秒人声片段，large-v3/float16）：模型加载 3.9s，
> 纯转写 6.7s → **10.4x 实时**。

## 目录结构

```
D:\tools\v2t\
├── v2t.cmd                 ← 命令行入口
├── v2t-webui.cmd           ← 图形界面入口（无窗口）
├── v2t-webui-debug.cmd     ← 图形界面入口（带控制台）
├── bin\
│   ├── v2t.py              主程序（run() 可被 WebUI 直接调用）
│   ├── webui.py            浏览器界面 + HTTP 服务
│   ├── bili_auth.py        B站登录/登出/状态（扫码、cookie 有效期管理）
│   ├── bili_subs.py        B站原生 AI 字幕直取（需登录）
│   ├── browser_fetch.py    浏览器嗅探抓取（抖音/快手/小红书）
│   ├── qwen_engine.py      Qwen3-ASR 适配器（跑在独立 venv 子进程）
│   ├── make_icon.py        生成 assets\icon.ico
│   ├── create_shortcut.py  重建桌面快捷方式
│   ├── fetch_model.ps1     下载 whisper 模型（走 hf-mirror）
│   └── fetch_cuda.py       下载 CUDA 运行库
├── assets\icon.ico         图标（10 个尺寸，小尺寸单独绘制）
├── data\                   配置 / 历史 / 日志
├── pylib\                  yt-dlp + opencc + CUDA dll（不污染 C 盘）
├── models\                 whisper 模型
│   ├── faster-whisper-small\    461 MB
│   ├── faster-whisper-medium\   1.46 GB
│   ├── faster-whisper-large-v3\ 2.95 GB
│   └── Qwen3-ASR-1.7B\          4.4 GB（中文主力）
├── venv_qwen\              Qwen3-ASR 独立环境（torch 2.9.1+cu126，5.0 GB）
└── wheels\                 CUDA wheel 缓存
```

## 已知限制

- **抖音/快手/小红书**：已支持，走浏览器嗅探（`bin/browser_fetch.py`）。
  代价是每个视频多等 5~10 秒开浏览器；高频批量可能触发滑块，此时加 `--headed` 手动过。
- **微信视频号**：**不支持**。视频号需要 PC 微信开着 + 本地代理 + 根证书，
  且视频加密（前 128KB 用 ISAAC-64），做不到「丢链接就行」。建议存手机相册再拖进来。
- **B站字幕**：走 `v2t login` 登录后直取官方 AI 字幕（推荐）。
  若用 `--cookies-from-browser`，注意浏览器运行时 cookie 库被锁，需先完全退出浏览器。
- **YouTube**：字幕能不能拿到**取决于视频本身**——很多中文搬运视频压根没有字幕
  （上传者没传、平台也没自动生成），此时只能 ASR。另外可能遇到
  `Sign in to confirm you're not a bot`（IP 级风控，需挂代理）。
- **纯音乐/BGM 段落**：会被静音检测剔除，这是有意的（避免幻听出歌词）。
- **同音字错误**：ASR 必然有。中文人名、专有名词、方言尤其容易错。要精确引用时请回视频核对。
  这也是为什么 B站视频强烈建议登录——直取官方字幕可完全避免这类错误。

## 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| B站视频走了 ASR、术语错得离谱 | 未登录。跑 `v2t login`，之后 1 秒出稿 |
| `[bili] 直取失败：cookie 已失效` | 180 天到期了，重新 `v2t login` |
| `Library cublas64_12.dll is not found` | **已修**：光 `os.add_dll_directory` 不够，ctranslate2 走系统 LoadLibrary 搜索顺序，必须把 `pylib\nvidia\*\bin` 也塞进 `PATH`。若仍报错，说明 dll 真缺，跑 `bin\fetch_cuda.py` |
| 抖音报 `Fresh cookies ... needed` | 已改走浏览器通道，不该再出现。若出现说明 playwright 没装好 |
| 浏览器抓取报「没抓到媒体直链」 | 视频可能私密/付费/需登录。加 `--headed` 看页面发生了什么 |
| 抓取时卡在验证码 | 加 `--headed`，在弹出窗口里手动过滑块 |
| 转写返回 0 段且极快 | 静音检测把音频全滤了。加 `--no-vad` 试试，或调 `--vad-noise-db -45` |
| `SSL: UNEXPECTED_EOF_WHILE_READING` | 网络抖动，已内置重试；持续失败就换网络或挂代理 `--proxy` |
| 输出繁体字 | 已用 `initial_prompt` + opencc 处理；若仍有，装 `opencc-python-reimplemented` |
