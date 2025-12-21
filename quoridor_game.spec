# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Quoridor Game
Optimized for lightweight AlphaBeta AI-only build
"""

import os
project_root = os.path.abspath(os.path.dirname(__name__))

block_cipher = None

# Analysis: Define what to include and exclude
a = Analysis(
    ['main.py'],
    pathex=[project_root],
    binaries=[],
    datas=[
        ('assets', 'assets'),  # Include the assets folder (tile.png)
    ],
    hiddenimports=[
        'numpy',
        'pygame',
        'AI',
        'AI.alpha_beta',
        'AI.action_utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude PyTorch and neural network AI components
        'torch',
        'torchvision',
        'torchaudio',
        'AI.network',
        'AI.alpha_mcts',
        'AI.encoder',
        # Exclude training and distributed components
        'AI.train',
        'AI.distributed',
        'tensorboard',
        'ray',
        'tqdm',
        'lightning',
        'lightning-sdk',
        # Exclude unnecessary standard library modules to reduce size
        'unittest',
        'test',
        'distutils',
        'setuptools',
        'pip',
        'wheel',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# PYZ: Create a Python archive
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

# EXE: Create the executable
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
    upx=True,  # Enable UPX compression
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Windowed mode (no console)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Best bytecode optimization level
    optimize=2,
)
