@echo off
REM ============================================================
REM  v2t WebUI - background launcher (no console window)
REM  Log: <v2t dir>\data\webui.log
REM  For troubleshooting use v2t-webui-debug.cmd (keeps console)
REM ============================================================
setlocal
set "V2T_HOME=%~dp0"
set "PYW=pythonw.exe"
where pythonw.exe >nul 2>&1 || set "PYW=python.exe"

start "" "%PYW%" "%~dp0bin\webui.py" --port 7788 %*
endlocal
