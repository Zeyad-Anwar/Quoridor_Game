# Quick Start: Release Quoridor Game as Executable

## 📋 Overview
This guide shows you how to automatically build Windows, Linux, and macOS executables using GitHub Actions and the `uv` package manager.

## ✅ What's Been Set Up

I've created the following files for you:

1. **`quoridor.spec`** - PyInstaller configuration for building executables
2. **`.github/workflows/release.yml`** - GitHub Actions workflow for automated builds
3. **`build.py`** - Local build script for testing
4. **`RELEASE.md`** - Detailed release documentation
5. **`pyproject.toml`** - Updated with build dependencies

## 🚀 Step-by-Step Release Process

### Step 1: Test Locally (Optional but Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the build script
python build.py

# Test the executable in dist/
./dist/Quoridor  # or Quoridor.exe on Windows
```

### Step 2: Commit All Changes

```bash
git add .
git commit -m "Add executable build configuration"
git push origin main
```

### Step 3: Create and Push a Release Tag

```bash
# Update version in pyproject.toml first if needed
# Then create a tag matching that version
git tag v0.2.0
git push origin v0.2.0
```

### Step 4: Wait for GitHub Actions

1. Go to your repository on GitHub
2. Click the "Actions" tab
3. Watch the "Build and Release Executables" workflow run
4. It will build for Windows, Linux, and macOS simultaneously

### Step 5: Download Your Release

1. Go to the "Releases" page of your repository
2. Click on the latest release (v0.2.0)
3. Download the appropriate file:
   - **Windows**: `Quoridor-Windows.zip`
   - **Linux**: `Quoridor-Linux.tar.gz`
   - **macOS**: `Quoridor-macOS.zip`

## 📝 Quick Reference

### Trigger a New Release

```bash
# 1. Update version in pyproject.toml
# 2. Commit changes
git add pyproject.toml
git commit -m "Bump version to v0.3.0"

# 3. Create and push tag
git tag v0.3.0
git push origin main
git push origin v0.3.0
```

### Manual Trigger (No Release)

1. Go to GitHub → Actions → "Build and Release Executables"
2. Click "Run workflow"
3. This builds executables but doesn't create a release

## 🎯 How It Works

### The Workflow:

1. **Triggered by**: Pushing a tag like `v*` (e.g., v1.0.0, v0.2.1)
2. **Runs on**: 3 platforms simultaneously (Ubuntu, Windows, macOS)
3. **For each platform**:
   - Installs `uv`
   - Sets up Python 3.12
   - Installs dependencies with `uv pip install`
   - Runs PyInstaller with your `quoridor.spec` configuration
   - Packages the executable (zip/tar.gz)
   - Uploads as artifact
4. **Creates GitHub Release** with all 3 executables attached

### Technologies Used:

- **uv**: Fast Python package manager (replaces pip/venv)
- **PyInstaller**: Bundles Python apps into standalone executables
- **GitHub Actions**: CI/CD automation
- **Multi-platform builds**: Windows, Linux, macOS

## 🔧 Customization

### Add an Icon

1. Create icon files:
   - Windows: `icon.ico`
   - macOS: `icon.icns`
   
2. Update `quoridor.spec`:
   ```python
   icon='assets/icon.ico'  # or assets/icon.icns for macOS
   ```

### Include Additional Files

Edit `quoridor.spec`, in the `datas` section:
```python
datas=[
    ('assets', 'assets'),
    ('checkpoints', 'checkpoints'),  # Add this if needed
    ('config.json', '.'),  # Add config file
],
```

### Change Executable Name

In `quoridor.spec`:
```python
name='YourGameName',  # Change from 'Quoridor'
```

### Enable Console Window (for debugging)

In `quoridor.spec`:
```python
console=True,  # Change from False
```

## ⚠️ Common Issues

### 1. Missing Python modules in executable
**Fix**: Add to `hiddenimports` in `quoridor.spec`:
```python
hiddenimports=[
    'pygame',
    'torch',
    'numpy',
    'your_missing_module',  # Add here
],
```

### 2. Assets not found
**Fix**: Ensure assets are in `datas` list and update code to use PyInstaller's resource path:
```python
import sys
import os

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Use it:
tile_img = pygame.image.load(resource_path("assets/tile.png"))
```

### 3. Workflow doesn't trigger
**Check**:
- Tag format matches `v*` pattern
- Tag was pushed: `git push origin v0.2.0`
- Repository has Actions enabled

### 4. Large executable size
**Optimize**:
- Enable UPX: Already enabled in spec (`upx=True`)
- Exclude unused modules in `quoridor.spec`
- Consider using `--onedir` instead of `--onefile` (modify spec)

## 📚 Next Steps

1. **Test the workflow**: Create a test tag and verify builds work
2. **Add icon**: Create and add icon files
3. **Code signing**: For production, consider signing executables
4. **Update README**: Add download links to releases

## 🔗 Resources

- [Full Release Guide](RELEASE.md) - Detailed documentation
- [PyInstaller Docs](https://pyinstaller.org/en/stable/)
- [uv Docs](https://docs.astral.sh/uv/)
- [GitHub Actions Docs](https://docs.github.com/en/actions)

---

**Ready to release?** Just run:
```bash
git tag v0.2.0 && git push origin v0.2.0
```
