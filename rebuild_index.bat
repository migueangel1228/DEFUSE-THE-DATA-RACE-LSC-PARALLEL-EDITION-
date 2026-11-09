@echo off
title Reconstruir Indice LSC
cd /d "%~dp0"
set "PYTHON_EXE=C:\Users\Usuario\AppData\Local\Programs\Python\Python313\python.exe"
if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" src\indexer.py
) else (
    python src\indexer.py
)
pause
