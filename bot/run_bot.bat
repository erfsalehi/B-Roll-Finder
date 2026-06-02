@echo off
setlocal
:: Launch the B-Roll Telegram bot using the project venv.
:: Run this from anywhere; it cd's to the project root (parent of this file).

cd /d "%~dp0\.."

if not exist venv (
    echo [ERROR] venv not found. Run run.bat once first to create it.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

:: TUN-VPN users: strip stale proxy vars so requests use the OS network layer.
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=

:: Auto-restart loop — if the bot ever crashes (network blip, etc.) it relaunches
:: after a short pause instead of staying down.
:loop
echo [INFO] Starting B-Roll Telegram bot...
python -m bot.telegram_bot
echo [WARN] Bot exited (code %errorlevel%). Restarting in 10s... (close this window to stop)
timeout /t 10 /nobreak >nul
goto loop
