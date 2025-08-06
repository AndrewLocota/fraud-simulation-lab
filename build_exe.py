# First install: pip install pyinstaller

import subprocess
import sys
import os

def build_executable():
    """Build standalone executable"""
    
    # Install PyInstaller if not present
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Create spec file for better control
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('poppler', 'poppler'),
        ('requirements.txt', '.'),
    ],
    hiddenimports=['streamlit', 'augraphy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FraudSimulation',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""
    
    with open("fraud_simulation.spec", "w") as f:
        f.write(spec_content)
    
    # Build the executable
    print("Building executable...")
    subprocess.run([
        "pyinstaller", 
        "--onefile", 
        "--add-data", "poppler;poppler",
        "--hidden-import", "streamlit",
        "--hidden-import", "augraphy",
        "app.py"
    ])
    
    print("✅ Executable created in dist/ folder")

if __name__ == "__main__":
    build_executable() 