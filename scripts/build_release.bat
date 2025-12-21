@echo off
REM Build script for Quoridor Game - Windows
REM Creates a distributable executable using PyInstaller

setlocal enabledelayedexpansion

echo =========================================
echo Quoridor Game - Build Script
echo =========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    exit /b 1
)

REM Display Python version
for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [OK] Using: %PYTHON_VERSION%

REM Check if we're in the project root
if not exist "quoridor_game.spec" (
    echo [ERROR] quoridor_game.spec not found. Please run this script from the project root.
    exit /b 1
)

REM Check if assets folder exists
if not exist "assets" (
    echo [WARNING] assets folder not found
)

REM Install/upgrade build dependencies
echo.
echo Installing uv and build dependencies...

REM Check if uv is installed
where uv >nul 2>&1
if errorlevel 1 (
    echo Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo [ERROR] Failed to install uv
        exit /b 1
    )
)

uv sync --extra build --no-dev
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    exit /b 1
)

REM Clean previous builds
echo.
echo Cleaning previous builds...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist

REM Run PyInstaller
echo.
echo Building with PyInstaller (this may take a few minutes)...
uv run pyinstaller quoridor_game.spec --clean
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed
    exit /b 1
)

REM Verify build
echo.
if exist "dist\Quoridor.exe" (
    echo [OK] Build successful!
    echo.
    echo Executable details:
    dir dist\Quoridor.exe
    echo.
    
    for %%A in (dist\Quoridor.exe) do set FILE_SIZE=%%~zA
    set /a FILE_SIZE_MB=!FILE_SIZE! / 1048576
    echo [OK] Build artifact: dist\Quoridor.exe (Size: ~!FILE_SIZE_MB! MB)
    echo.
    echo To run the game:
    echo   dist\Quoridor.exe
) else (
    echo [ERROR] Build failed! Executable not found in dist\Quoridor.exe
    exit /b 1
)

echo.
echo =========================================
echo Build Complete
echo =========================================

endlocal
