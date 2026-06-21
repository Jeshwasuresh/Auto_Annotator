@echo off
title AI Auto Annotator v4
color 0B
cls
echo.
echo  =====================================================================
echo    AI Auto Annotator v4  ^|  Windows Edition
echo    Smart Train + Propagate  ^|  Session Save/Load  ^|  YOLOv8 Offline
echo  =====================================================================
echo.

:: ── Try pre-built exe first ──────────────────────────────────────────────────
if exist "%~dp0dist\AutoAnnotator.exe" (
    echo  [OK] Found AutoAnnotator.exe in dist  -  Launching...
    start "" "%~dp0dist\AutoAnnotator.exe"
    exit /b 0
)
if exist "%~dp0AutoAnnotator\AutoAnnotator.exe" (
    echo  [OK] Found AutoAnnotator.exe  -  Launching...
    start "" "%~dp0AutoAnnotator\AutoAnnotator.exe"
    exit /b 0
)
if exist "%~dp0AutoAnnotator.exe" (
    start "" "%~dp0AutoAnnotator.exe"
    exit /b 0
)

:: ── Fall back to Python source ───────────────────────────────────────────────
echo  [INFO] No .exe found  -  running from Python source...
echo.

set PY=
where python  >nul 2>&1 && set PY=python
if "%PY%"=="" (where python3 >nul 2>&1 && set PY=python3)
if "%PY%"=="" (
    echo  [ERROR] Python not found!
    echo.
    echo  Please install Python 3.9+ from:  https://python.org/downloads
    echo  IMPORTANT: tick "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

echo  [INFO] Checking packages...
%PY% -c "import customtkinter,ultralytics,cv2,PIL" >nul 2>&1
if errorlevel 1 (
    echo  [INFO] Installing packages (first time only, needs internet)...
    echo.
    %PY% -m pip install customtkinter ultralytics opencv-python pillow --quiet
    if errorlevel 1 (
        echo  [ERROR] Package install failed. Try running as Administrator.
        pause & exit /b 1
    )
)

echo  [OK] All packages ready.
echo  [>>] Starting AI Auto Annotator v4...
echo.
%PY% "%~dp0src\main.py"
if errorlevel 1 (
    echo.
    echo  [ERROR] App crashed. See message above.
    pause
)
