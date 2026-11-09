@echo off
title Defuse the Data-Race: LSC Parallel Edition
cd /d "%~dp0"
set "PYTHON_EXE=C:\Users\Usuario\AppData\Local\Programs\Python\Python313\python.exe"
if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" src\game_ui.py
) else (
    python src\game_ui.py
)
if %errorlevel% neq 0 (
    echo.
    echo Ocurrio un error al ejecutar el juego.
    pause
)
