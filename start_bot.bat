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

:: NOTE: we intentionally do NOT clear HTTP_PROXY/HTTPS_PROXY here — if Telegram
:: is reachable only through your VPN's local proxy, clearing it breaks the bot.
:: TUN-mode users who hit stale-proxy errors should set BROLL_BYPASS_HTTP_PROXY=1
:: in .env instead (the bot honors it). To force a proxy, set BOT_PROXY in .env.

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
