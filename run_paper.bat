@echo off
REM One-click daily cycle: data -> fills -> scoring -> picks -> Alpaca paper orders.
REM (The scheduled task StockAI_Daily runs this automatically every weekday 22:00 -
REM  this file is for running it manually whenever you like.)
cd /d "%~dp0"
.venv\Scripts\python.exe run_daily.py
pause
