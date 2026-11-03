@echo off
REM v2t WebUI - debug launcher (keeps console, shows errors)
setlocal
set "V2T_HOME=%~dp0"
set "PYTHON=python"
echo Starting v2t WebUI...
"%PYTHON%" "%~dp0bin\webui.py" --port 7788 %*
echo.
echo Server exited. Press any key to close.
pause >nul
endlocal
