@echo off
title Build AI Auto Annotator v4 - EXE
color 0A
cls
echo.
echo  ============================================================
echo    AI Auto Annotator v4  ^|  Windows EXE Builder
echo  ============================================================
echo.
echo  Estimated time: 4-10 minutes on first build.
echo.

:: Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install from https://python.org
    echo  Tick "Add Python to PATH" during setup.
    pause & exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo  [OK] %%v

echo.
echo  [1/4] Installing / upgrading packages...
python -m pip install --upgrade --quiet ^
    customtkinter ultralytics opencv-python pillow pyinstaller
echo  [OK] Packages ready.
echo.

echo  [2/4] Cleaning old build files...
if exist build rmdir /s /q build >nul 2>&1
if exist dist  rmdir /s /q dist  >nul 2>&1
echo  [OK] Clean done.
echo.

echo  [3/4] Running PyInstaller (this takes a few minutes)...
python -m PyInstaller ^
    --clean --noconfirm ^
    --name AutoAnnotator ^
    --windowed ^
    --onefile ^
    --hidden-import customtkinter ^
    --hidden-import PIL._tkinter_finder ^
    --hidden-import PIL.Image ^
    --hidden-import PIL.ImageTk ^
    --hidden-import ultralytics ^
    --hidden-import ultralytics.nn.tasks ^
    --hidden-import ultralytics.utils ^
    --hidden-import ultralytics.models.yolo ^
    --hidden-import cv2 ^
    --hidden-import numpy ^
    --hidden-import tkinter ^
    --hidden-import tkinter.filedialog ^
    --hidden-import tkinter.messagebox ^
    --hidden-import tkinter.simpledialog ^
    --hidden-import json ^
    --hidden-import threading ^
    --hidden-import shutil ^
    --hidden-import tempfile ^
    --collect-all customtkinter ^
    --collect-all ultralytics ^
    --distpath dist ^
    --workpath build ^
    src\main.py

if errorlevel 1 (
    echo.
    echo  [ERROR] Build failed. Check the output above for details.
    pause & exit /b 1
)

echo.
echo  [4/4] Finalizing build...
echo.
echo  ============================================================
echo   BUILD COMPLETE!
echo.
echo   Executable : dist\AutoAnnotator.exe
echo.
echo   To share:  simply copy or send dist\AutoAnnotator.exe!
echo   Recipient can double-click and run it directly.
echo  ============================================================
echo.
pause
