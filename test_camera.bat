@echo off
title Diagnostico y Prueba de Camaras LSC
cd /d "%~dp0"
set "PYTHON_EXE=C:\Users\Usuario\AppData\Local\Programs\Python\Python313\python.exe"
if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" test_camera.py
) else (
    python test_camera.py
)
pause
