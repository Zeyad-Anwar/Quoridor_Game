#!/bin/bash
# Build script for Quoridor Game - Linux/macOS
# Creates a distributable binary using PyInstaller

set -e  # Exit on error

echo "========================================="
echo "Quoridor Game - Build Script"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed or not in PATH${NC}"
    exit 1
fi

# Display Python version
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}Using: $PYTHON_VERSION${NC}"

# Check if we're in the project root
if [ ! -f "quoridor_game.spec" ]; then
    echo -e "${RED}Error: quoridor_game.spec not found. Please run this script from the project root.${NC}"
    exit 1
fi

# Check if assets folder exists
if [ ! -d "assets" ]; then
    echo -e "${YELLOW}Warning: assets folder not found${NC}"
fi

# Install/upgrade build dependencies
echo ""
echo "Installing uv and build dependencies..."
# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --extra build --no-dev

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build dist

# Run PyInstaller
echo ""
echo "Building with PyInstaller (this may take a few minutes)..."
uv run pyinstaller quoridor_game.spec --clean

# Verify build
echo ""
if [ -f "dist/Quoridor" ]; then
    echo -e "${GREEN}Build successful!${NC}"
    echo ""
    echo "Executable details:"
    ls -lh dist/Quoridor
    echo ""
    
    # Make executable if not already
    chmod +x dist/Quoridor
    
    FILE_SIZE=$(du -h dist/Quoridor | cut -f1)
    echo -e "${GREEN}Build artifact: dist/Quoridor (Size: $FILE_SIZE)${NC}"
    echo ""
    echo "To run the game:"
    echo "  ./dist/Quoridor"
else
    echo -e "${RED}Build failed! Executable not found in dist/Quoridor${NC}"
    exit 1
fi

echo ""
echo "========================================="
echo "Build Complete"
echo "========================================="
