@echo off
REM v2t -- video URL to transcript (subtitle first, local ASR fallback)
REM Usage: v2t.cmd "https://www.bilibili.com/video/BVxxxx"
REM        v2t.cmd login      login to Bilibili (native AI subtitles, ~180 days)
REM        v2t.cmd status     show login status
REM        v2t.cmd logout     clear saved login
setlocal
chcp 65001 >nul 2>&1
set "V2T_HOME=%~dp0"
set "PYTHONPATH=%~dp0pylib"
set PYTHONIOENCODING=utf-8
python "%~dp0bin\v2t.py" %*
endlocal
