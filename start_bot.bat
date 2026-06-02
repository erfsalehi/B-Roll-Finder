@echo off
title B-Roll Telegram Bot
cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment not found.
    echo Run run.bat once first to set up the project, then double-click this again.
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

:: TUN-VPN users: clear stale proxy vars so requests use the OS network layer.
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=

echo ============================================================
echo   B-Roll Telegram Bot
echo   Keep this window OPEN while you want the bot running.
echo   Close this window (or press Ctrl+C) to stop the bot.
echo ============================================================
echo.

:loop
python -m bot.telegram_bot
echo.
echo [WARN] Bot stopped (exit code %errorlevel%). Restarting in 10s...
echo        Close this window to stop for good.
timeout /t 10 /nobreak >nul
goto loop
