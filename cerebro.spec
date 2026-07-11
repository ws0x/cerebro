# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files

# trafilatura reads settings.cfg at runtime via configparser (not a Python
# import), so PyInstaller's static import analysis never sees it -- omitting
# it doesn't raise ImportError, it makes every fetch silently fail with
# "No option 'download_timeout' in section: 'DEFAULT'" instead. Found by
# actually building and running the frozen exe against a local test server,
# not just checking that dependencies import.
datas = collect_data_files('trafilatura')

a = Analysis(
    ['build_scripts\\entry.py'],
    pathex=['src'],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['faster_whisper', 'ctranslate2', 'av', 'tokenizers'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='cerebro',
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
