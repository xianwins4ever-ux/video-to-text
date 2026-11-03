# 贡献指南

感谢愿意改进这个项目。下面是几条约定，能让改动更快被合并。

## 开发环境

```powershell
git clone https://github.com/xianwins4ever-ux/video-to-text.git
cd video-to-text

# 主程序依赖（轻量）
pip install yt-dlp opencc-python-reimplemented

# 可选：浏览器嗅探通道（抖音/快手/小红书）
pip install playwright
playwright install msedge

# ffmpeg 任选一种装法
winget install Gyan.FFmpeg
# 或解压到 <项目目录>/ffmpeg/
# 或设置 V2T_FFMPEG_DIR
```

验证环境：

```powershell
python bin/paths.py     # 看路径探测结果
v2t status              # 看 B站登录态
```

## 铁律

### 1. 不要硬编码本机绝对路径

所有路径从 `bin/paths.py` 取：

```python
import paths
paths.HOME      # 项目根
paths.MODELS    # 模型目录
paths.ffmpeg_exe()
paths.cookie_dir()
```

需要新路径就加进 `paths.py`，**不要**在业务代码里写 `Path(r"D:\...")`。

### 2. 不要提交这些东西

| 路径 | 原因 |
|---|---|
| `models/` | 9GB+ 模型，用 `bin/fetch_models.py` 重新下载 |
| `venv_qwen/` | 5GB 虚拟环境 |
| `pylib/` | 内置依赖，可由 pip 重装 |
| `wheels/` | CUDA wheel 缓存 |
| `~/.v2t/cookies/` | **登录凭证，泄露等于账号被盗** |

已在 `.gitignore` 里排除，但提交前请自己再确认一遍：

```powershell
git status --short
```

### 3. B站相关改动必须测两种状态

B站字幕走登录态，登录与否是两条完全不同的代码路径：

```powershell
v2t status                    # 确认已登录
v2t "<bili-url>"              # 应走 bilibili_native，1秒出稿

v2t logout
v2t "<bili-url>"              # 应给出明确提示后降级 ASR
v2t login                     # 记得登回来
```

### 4. 提交信息用中文，说清"为什么"

```
✅ 修复 B站字幕 CDN 偶发断连导致转写失败
   aisubtitle.hdslb.com 约 1/4 概率返回 Remote end closed connection，
   加 3 次退避重试。

❌ update v2t.py
❌ fix bug
```

## 如果遇到平台风控

**先看 README 的「平台支持」章节**，那里记录了实测结论：

- YouTube 多数中文视频**真的没字幕**（不是权限问题）
- 抖音关键词搜索是**平台封闭**，连 MediaCrawler 也失败

这些已经花了很多时间验证，重复尝试没有意义。如果你发现了新的可行路径，
请在 issue 里附上**可复现的证据**（截图、接口返回原文），我们再一起看。

## 代码风格

- 遵循 PEP 8，但不必教条
- 注释解释**为什么**，不是**做什么**（代码本身能说明做了什么）
- 拿不准的地方留 `# TODO: 原因`，不要留空注释
- 中文项目，注释和日志用中文

## 提 PR 之前

```powershell
# 1. 语法自检
python -c "import ast,glob; [ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('bin/*.py')]"

# 2. 功能自测（至少跑通一条 B站链接）
v2t "https://www.bilibili.com/video/BV1fxni6XEwD/"

# 3. 确认没有敏感文件混入
git status --short
```

PR 描述请按模板填，「怎么验证的」那一栏**不要写"测过了"**，写具体命令和输出。
