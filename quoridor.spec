# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'),  # Include assets folder
    ],
    hiddenimports=[
        'pygame',
        'torch',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Quoridor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to False for windowed mode (no console)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add path to .ico file for Windows or .icns for macOS if you have one
)

# For macOS, create an app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='Quoridor.app',
        icon=None,  # Add path to .icns file if you have one
        bundle_identifier='com.quoridor.game',
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': 'True',
        },
    )
