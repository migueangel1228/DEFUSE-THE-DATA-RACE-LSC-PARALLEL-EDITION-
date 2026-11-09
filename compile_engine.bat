@echo off
title Compilar Motor C++ OpenMP
cd /d "%~dp0"
set "PYTHON_EXE=C:\Users\Usuario\AppData\Local\Programs\Python\Python313\python.exe"
if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" src\compile_engine.py
) else (
    python src\compile_engine.py
)
pause
