# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for HL Baker.   Build with:   pyinstaller --noconfirm hlbake.spec
# Produces a one-folder app in dist/HLBaker containing two executables:
#   HLBaker[.exe]  - the desktop app          hlbake[.exe] - command line + background worker
# (plus dist/HLBaker.app on macOS).
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None
datas, binaries, hiddenimports = [], [], []

# --- Blender (bpy) : the whole package incl. its data folder (scripts, Cycles kernels, ...)
for pkg in ("bpy", "numpy", "zstandard", "cattrs", "attrs", "attr", "requests", "urllib3", "certifi",
            "idna", "charset_normalizer", "OpenGL"):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass

# Blender's bundled Python scripts (importers, Cycles add-on) are loaded at runtime and are not
# seen by PyInstaller's analysis, so ship the full standard library they may need.
SKIP = {"tkinter", "turtle", "turtledemo", "idlelib", "test", "lib2to3", "ensurepip", "venv",
        "pydoc_data", "antigravity", "this", "__phello__", "_tkinter"}
for name in sorted(getattr(sys, "stdlib_module_names", ())):
    if name in SKIP or name.startswith("_test"):
        continue
    try:
        __import__(name)
    except Exception:
        continue
    hiddenimports.append(name)
    try:
        hiddenimports += [m for m in collect_submodules(name) if ".test" not in m]
    except Exception:
        pass

hiddenimports += collect_submodules("hlbake")
datas += [("icon.png", "."), ("blender_addon/hl_baker.py", "blender_addon")]

# Windows: Qt's software OpenGL renderer, used when the GPU driver lacks OpenGL 3.3
if sys.platform == "win32":
    import os as _os, PySide6 as _ps
    _sw = _os.path.join(_os.path.dirname(_ps.__file__), "opengl32sw.dll")
    if _os.path.exists(_sw):
        binaries += [(_sw, "PySide6"), (_sw, ".")]

a = Analysis(
    ["hlbake_main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pkg_resources", "setuptools", "PySide6.QtWebEngineCore", "PySide6.Qt3DCore",
              "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtPdf"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

icon = "icon.png" if sys.platform in ("win32", "darwin") else None   # converted by PyInstaller
# Windowed app (no console window on Windows/macOS) ...
exe_gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name="HLBaker", debug=False,
              strip=False, upx=False, console=False, icon=icon)
# ... and a console executable for the command line; the app also runs it (hidden) as its worker.
exe_cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name="hlbake", debug=False,
              strip=False, upx=False, console=True, icon=icon)
coll = COLLECT(exe_gui, exe_cli, a.binaries, a.zipfiles, a.datas, strip=False, upx=False,
               name="HLBaker")

if sys.platform == "darwin":
    app = BUNDLE(coll, name="HLBaker.app", icon=icon, bundle_identifier="org.hlbaker.app",
                 info_plist={"NSHighResolutionCapable": True, "CFBundleShortVersionString": "1.2.0",
                             "CFBundleDocumentTypes": [{"CFBundleTypeName": "HL Baker Project",
                                                        "CFBundleTypeExtensions": ["hlbake"],
                                                        "CFBundleTypeRole": "Editor"}]})

# --- post-build fix-up -------------------------------------------------------------------------
# OpenImageDenoise loads its device plugins from the folder its core library is loaded from.
# PyInstaller links the core library into the bundle root, so put the plugins there as well.
import os, shutil, glob
_roots = [os.path.join(DISTPATH, "HLBaker", "_internal"),
          os.path.join(DISTPATH, "HLBaker.app", "Contents", "Frameworks")]
for _internal in [r for r in _roots if os.path.isdir(r)]:
    _libdir = None
    for cand in ("bpy/lib", "bpy"):
        if glob.glob(os.path.join(_internal, cand, "*OpenImageDenoise_device*")):
            _libdir = os.path.join(_internal, cand)
            break
    if not _libdir:
        continue
    for src in glob.glob(os.path.join(_libdir, "*OpenImageDenoise_device*")):
        dst = os.path.join(_internal, os.path.basename(src))
        if os.path.lexists(dst):
            continue
        try:
            os.symlink(os.path.relpath(src, _internal), dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)
