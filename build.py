#!/usr/bin/env python3
"""
Build script for creating executable distributions of Quoridor Game.
This script uses uv and PyInstaller to build the application.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd, shell=False):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=shell)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        sys.exit(result.returncode)


def main():
    """Main build function."""
    print("=" * 60)
    print("Quoridor Game - Executable Build Script")
    print("=" * 60)

    # Get project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Check if uv is installed
    print("\n1. Checking for uv...")
    uv_check = subprocess.run(["uv", "--version"], capture_output=True)
    if uv_check.returncode != 0:
        print("Error: uv is not installed. Please install it first:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    print(f"   ✓ uv is installed: {uv_check.stdout.decode().strip()}")

    # Create virtual environment
    print("\n2. Creating virtual environment...")
    if not Path(".venv").exists():
        run_command(["uv", "venv"])
    else:
        print("   ✓ Virtual environment already exists")

    # Install dependencies
    print("\n3. Installing dependencies...")
    run_command(["uv", "pip", "install", "pyinstaller"])
    run_command(["uv", "pip", "install", "-e", "."])

    # Activate virtual environment and run PyInstaller
    print("\n4. Building executable with PyInstaller...")

    system = platform.system()
    if system == "Windows":
        activate_cmd = ".venv\\Scripts\\activate"
        pyinstaller_cmd = ".venv\\Scripts\\pyinstaller"
    else:
        activate_cmd = "source .venv/bin/activate"
        pyinstaller_cmd = ".venv/bin/pyinstaller"

    # Clean previous builds
    for dir_name in ["build", "dist"]:
        if Path(dir_name).exists():
            print(f"   Cleaning {dir_name}/...")
            shutil.rmtree(dir_name)

    # Run PyInstaller
    run_command([pyinstaller_cmd, "quoridor.spec"])

    print("\n" + "=" * 60)
    print("Build Complete!")
    print("=" * 60)
    print(f"Executable location: {project_root / 'dist'}")

    # Display the contents
    dist_path = project_root / "dist"
    if dist_path.exists():
        print("\nGenerated files:")
        for item in dist_path.iterdir():
            print(f"  - {item.name}")

    print("\nYou can now run the executable from the dist/ directory.")


if __name__ == "__main__":
    main()
