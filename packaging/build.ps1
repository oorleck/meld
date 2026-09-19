# Builds the Windows app and installer.   powershell -File packaging\build.ps1
#
#   1. installs the build tools (PyInstaller) without touching your other packages
#   2. draws the icon and downloads the current yt-dlp to ship inside the app
#   3. freezes the app into dist\Meld\ with PyInstaller
#   4. self-tests the frozen app (sync, mixing, video cutting, YouTube search)
#   5. wraps it into dist-installer\Meld-Setup-<version>.exe with Inno Setup (if installed:
#      https://jrsoftware.org/isdl.php)
#
# Skip the slow-changing parts with -SkipTest.
param([switch]$SkipTest)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

uv sync --inexact --group build
if ($LASTEXITCODE) { throw "uv sync failed" }

uv run --no-sync python packaging/make_icon.py
uv run --no-sync python -c "from pathlib import Path; from meld import ytdlp; print('bundling yt-dlp', ytdlp.download_to(Path('packaging/build/yt-dlp.zip')))"
if ($LASTEXITCODE) { throw "could not download yt-dlp (are you online?)" }

uv run --no-sync pyinstaller packaging/meld.spec --noconfirm --distpath dist --workpath packaging/build/work
if ($LASTEXITCODE) { throw "PyInstaller failed" }

if (-not $SkipTest) {
    $report = Join-Path $env:TEMP "meld-selftest.txt"
    Remove-Item $report -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath "dist\Meld\Meld.exe" -ArgumentList "--selftest", $report, "--online" -Wait -PassThru
    Get-Content $report
    if ($p.ExitCode -ne 0) { throw "The frozen app failed its self-test (exit code $($p.ExitCode))" }
}

$version = (uv run --no-sync python -c "import meld; print(meld.__version__)").Trim()
$iscc = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
# Installs the fresh installer into a scratch folder, checks every file arrived, self-tests it and uninstalls.
# (An installer can come out corrupt without any error at build time; this catches that.)
function Test-Installer([string]$setup, [string]$distDir) {
    $dir = Join-Path $env:TEMP "Meld-verify"
    if (Test-Path $dir) { [IO.Directory]::Delete($dir, $true) }
    $p = Start-Process -FilePath $setup -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER", "/DIR=$dir" -Wait -PassThru
    if ($p.ExitCode -ne 0 -or -not (Test-Path "$dir\Meld.exe")) { return $false }
    $want = (Get-ChildItem $distDir -Recurse -File | Measure-Object).Count
    $got = (Get-ChildItem $dir -Recurse -File | Measure-Object).Count - 2   # minus the uninstaller's own two files
    $ok = ($got -eq $want)
    if ($ok -and -not $SkipTest) {
        $report = Join-Path $env:TEMP "meld-selftest-installed.txt"
        $t = Start-Process -FilePath "$dir\Meld.exe" -ArgumentList "--selftest", $report -Wait -PassThru
        $ok = ($t.ExitCode -eq 0)
    }
    Start-Process -FilePath "$dir\unins000.exe" -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -Wait | Out-Null
    Start-Sleep -Seconds 2
    if (Test-Path $dir) { [IO.Directory]::Delete($dir, $true) }
    return $ok
}

if ($iscc) {
    $setup = "dist-installer\Meld-Setup-$version.exe"
    $verified = $false
    for ($try = 1; $try -le 2 -and -not $verified; $try++) {
        & $iscc "/DAppVersion=$version" packaging\installer.iss
        if ($LASTEXITCODE) { throw "Inno Setup failed" }
        $verified = Test-Installer (Resolve-Path $setup).Path (Resolve-Path "dist\Meld").Path
        if (-not $verified) { Write-Warning "The installer failed verification (attempt $try); rebuilding it." }
    }
    if (-not $verified) { throw "The installer failed verification twice. Do not distribute it." }
    Write-Host "`nInstaller (verified: installs, runs, uninstalls cleanly): $setup"
} else {
    Write-Host "`nInno Setup is not installed, so only the app folder was built: dist\Meld"
}
