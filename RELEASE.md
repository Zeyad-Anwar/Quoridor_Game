# Quoridor Game - Release Guide

This guide explains how to create executable releases of the Quoridor game using GitHub Actions.

## Prerequisites

- GitHub repository with the code pushed
- `uv` package manager installed locally (for local testing)
- Git installed locally

## Release Process

### Method 1: Automatic Release via Git Tags (Recommended)

1. **Update the version** in `pyproject.toml`:
   ```toml
   version = "0.2.0"  # Change to your desired version
   ```

2. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Prepare release v0.2.0"
   ```

3. **Create and push a version tag**:
   ```bash
   git tag v0.2.0
   git push origin main
   git push origin v0.2.0
   ```

4. **Wait for GitHub Actions** to build executables for:
   - Windows (x86_64)
   - Linux (x86_64)
   - macOS (x86_64)

5. **Check the release** at: `https://github.com/YOUR_USERNAME/YOUR_REPO/releases`

### Method 2: Manual Trigger

1. Go to your repository on GitHub
2. Click on "Actions" tab
3. Select "Build and Release Executables" workflow
4. Click "Run workflow" button
5. Select the branch and click "Run workflow"

Note: This will build the executables but won't create a release unless you're on a tagged commit.

## Local Testing

Before creating a release, test the build locally:

```bash
# Make the build script executable (Linux/macOS)
chmod +x build.py

# Run the build
python build.py
```

Or manually:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Install dependencies
uv pip install pyinstaller
uv pip install -e .

# Build with PyInstaller
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pyinstaller quoridor.spec
```

The executable will be in the `dist/` directory.

## Customizing the Build

### Modify PyInstaller Configuration

Edit `quoridor.spec` to:
- Add an icon: Set `icon='path/to/icon.ico'` (Windows) or `icon='path/to/icon.icns'` (macOS)
- Include additional data files in the `datas` list
- Add hidden imports if PyInstaller misses dependencies
- Change console mode: Set `console=True` for debugging

### Modify GitHub Actions Workflow

Edit `.github/workflows/release.yml` to:
- Build for specific platforms only
- Add code signing (for Windows/macOS)
- Run tests before building
- Add additional build steps

## Release Assets

Each release will include:

- **Linux**: `Quoridor-Linux.tar.gz` - Compressed executable for Linux
- **Windows**: `Quoridor-Windows.zip` - Compressed `.exe` for Windows
- **macOS**: `Quoridor-macOS.zip` - Compressed `.app` bundle for macOS

## Troubleshooting

### Build fails with missing modules
- Add the module to `hiddenimports` in `quoridor.spec`

### Assets (images, fonts) not found
- Ensure they're listed in the `datas` section of `quoridor.spec`
- Use resource paths correctly in your code (see PyInstaller documentation)

### Large executable size
- Enable UPX compression: `upx=True` in `quoridor.spec`
- Exclude unnecessary modules in the `excludes` list

### macOS "App is damaged" error
- This is due to unsigned code. Users need to:
  - Right-click the app → "Open" (first time only)
  - Or: System Preferences → Security & Privacy → Allow

### Windows SmartScreen warning
- This is normal for unsigned executables
- Users can click "More info" → "Run anyway"
- Consider code signing for production releases

## Advanced: Code Signing

### Windows
Add to the workflow:
```yaml
- name: Sign Windows executable
  run: signtool sign /f certificate.pfx /p ${{ secrets.CERT_PASSWORD }} dist/Quoridor.exe
```

### macOS
Add to the workflow:
```yaml
- name: Sign macOS app
  run: |
    codesign --force --deep --sign "${{ secrets.MACOS_CERTIFICATE }}" dist/Quoridor.app
    codesign --verify --verbose dist/Quoridor.app
```

## Version Bumping

For semantic versioning (e.g., v1.2.3):
- **Patch** (v1.2.3 → v1.2.4): Bug fixes, minor changes
- **Minor** (v1.2.3 → v1.3.0): New features, backward compatible
- **Major** (v1.2.3 → v2.0.0): Breaking changes

Update version in:
1. `pyproject.toml` - `version` field
2. Git tag when releasing

## Resources

- [PyInstaller Documentation](https://pyinstaller.org/en/stable/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
