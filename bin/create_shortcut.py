# -*- coding: utf-8 -*-
"""创建/重建桌面快捷方式「视频转文案」。

可重复运行（先删旧的再建，避免图标缓存住旧图标）。
用法：
    powershell -ExecutionPolicy Bypass -File create_shortcut.ps1
或直接：
    python create_shortcut.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# pythonw.exe 优先与当前解释器同目录；没有就退回 python.exe
_pyw = Path(sys.executable).with_name("pythonw.exe")
TARGET = _pyw if _pyw.exists() else Path(sys.executable)
SCRIPT = ROOT / "bin" / "webui.py"
ICON = ROOT / "assets" / "icon.ico"
DESKTOP = Path.home() / "Desktop"
LNK = DESKTOP / "视频转文案.lnk"

PS = f'''
$ErrorActionPreference = "Stop"
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut("{LNK}")
$lnk.TargetPath       = "{TARGET}"
$lnk.Arguments        = '"{SCRIPT}" --port 7788'
$lnk.WorkingDirectory = "{ROOT}"
$lnk.IconLocation     = "{ICON},0"
$lnk.Description      = "视频链接/文件 → 带时间戳的文案（本地转录，内容不出本机）"
$lnk.WindowStyle      = 7
$lnk.Save()
'''

if __name__ == "__main__":
    import subprocess
    if LNK.exists():
        LNK.unlink()
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                        "-Command", PS], capture_output=True, text=True)
    if r.returncode != 0:
        print("失败：", r.stderr[:500])
        raise SystemExit(1)
    print("已创建：", LNK)
    print("  目标：", TARGET)
    print("  参数：", SCRIPT, "--port 7788")
    print("  图标：", ICON)
