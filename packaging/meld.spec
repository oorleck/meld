# PyInstaller spec for the Windows app. Build with packaging/build.ps1 (or: pyinstaller packaging/meld.spec).
from pathlib import Path

root = Path(SPECPATH).parent
build = root / "packaging" / "build"

datas = [(str(root / "packaging" / "meld.ico"), ".")]
# yt-dlp is shipped as a plain zip, not frozen into the app, so a newer copy in the user's data folder can
# replace it (YouTube breaks old versions). See meld/ytdlp.py.
if (build / "yt-dlp.zip").exists():
    datas.append((str(build / "yt-dlp.zip"), "."))


def stdlib_modules_used_by(zip_path):
    """yt-dlp is not analysed (it is a zip), so find which standard-library modules it imports and bundle those."""
    import ast
    import sys
    import zipfile

    names = set()
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if not member.endswith(".py"):
                continue
            try:
                tree = ast.parse(z.read(member))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names.add(node.module)
    return sorted(n for n in names if n.split(".")[0] in sys.stdlib_module_names)


ytdlp_needs = stdlib_modules_used_by(build / "yt-dlp.zip") if (build / "yt-dlp.zip").exists() else []

a = Analysis(
    [str(root / "packaging" / "gui_entry.py")],
    pathex=[str(root / "src")],
    datas=datas,
    hiddenimports=ytdlp_needs,
    excludes=[
        "yt_dlp",  # shipped as yt-dlp.zip instead
        # the optional 3D reconstruction stack (several GB) is not part of the installed app
        "torch", "torchvision", "vggt", "einops", "safetensors", "huggingface_hub", "PIL",
        # things that get pulled in by accident
        "matplotlib", "pandas", "IPython", "pytest", "pygments", "jedi", "notebook",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Meld",
    icon=str(root / "packaging" / "meld.ico"),
    console=False,  # a normal window app, no black terminal
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Meld", upx=False)
