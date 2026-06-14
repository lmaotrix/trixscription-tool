# Install system dependencies and Python environment for Trixscription on Windows 11.
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

Write-Host "=== Trixscription Windows installer ===" -ForegroundColor Cyan

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Refresh-Path {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"
}

function Find-FFmpegExe {
    Refresh-Path

    $cmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $searchRoots = @(
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Packages"
    )
    foreach ($root in $searchRoots) {
        if (-not (Test-Path $root)) { continue }
        $found = Get-ChildItem $root -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($found) {
            return $found.FullName
        }
    }

    return $null
}

function Ensure-FFmpeg {
    $ffmpegExe = Find-FFmpegExe
    if ($ffmpegExe) {
        $binDir = Split-Path -Parent $ffmpegExe
        if ($env:Path -notlike "*$binDir*") {
            $env:Path = "$binDir;$env:Path"
        }
        Write-Host "FFmpeg found: $ffmpegExe"
        return
    }

    Write-Host "FFmpeg not found. Attempting install via winget..."
    if (-not (Test-Command winget)) {
        Write-Error "winget is unavailable. Install FFmpeg from https://ffmpeg.org/download.html and re-run."
    }

    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    Refresh-Path

    $ffmpegExe = Find-FFmpegExe
    if (-not $ffmpegExe) {
        Write-Error @"
FFmpeg was installed but could not be located.
Restart PowerShell and re-run this script, or add FFmpeg's bin folder to PATH manually.
"@
    }

    $binDir = Split-Path -Parent $ffmpegExe
    $env:Path = "$binDir;$env:Path"
    Write-Host "FFmpeg found: $ffmpegExe"
}

function Find-PythonLauncher {
    if (-not (Test-Command py)) {
        return $null
    }

    # Prefer 3.12 or 3.11 for package compatibility; avoid bleeding-edge versions.
    foreach ($version in @("3.12", "3.11", "3.13", "3")) {
        $probe = & py "-$version" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @{ Launcher = "py"; VersionFlag = "-$version"; Version = $probe }
        }
    }

    return $null
}

function Find-Python {
    $launcher = Find-PythonLauncher
    if ($launcher) {
        return $launcher
    }

    foreach ($candidate in @("python", "python3")) {
        if (-not (Test-Command $candidate)) { continue }
        $version = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @{ Launcher = $candidate; VersionFlag = $null; Version = $version }
        }
    }

    return $null
}

function Install-Python {
    if (-not (Test-Command winget)) {
        Write-Error "Python not found and winget is unavailable. Install Python 3.12 from https://www.python.org/downloads/"
    }

    Write-Host "Installing Python 3.12 via winget..."
    winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
    Refresh-Path
}

function Invoke-Python {
    param([hashtable]$PythonInfo, [string[]]$Arguments)

    if ($PythonInfo.Launcher -eq "py") {
        return & py $PythonInfo.VersionFlag @Arguments
    }
    return & $PythonInfo.Launcher @Arguments
}

$pythonInfo = Find-Python
if (-not $pythonInfo) {
    Install-Python
    Refresh-Path
    $pythonInfo = Find-Python
    if (-not $pythonInfo) {
        Write-Error "Python 3 not found after install attempt. Restart your terminal and re-run."
    }
}

Write-Host "Using Python $($pythonInfo.Version)"
if ($pythonInfo.Version -eq "3.14") {
    Write-Warning "Python 3.14 may be too new for some packages. Install Python 3.12 if pip installs fail."
}

Ensure-FFmpeg

Write-Host "Creating virtual environment..."
Invoke-Python $pythonInfo @("-m", "venv", "venv")

$venvPython = Join-Path $RootDir "venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Failed to create virtual environment."
}

& $venvPython -m pip install --upgrade pip

Write-Host "Installing PyTorch with CUDA 12.8 support..."
& $venvPython -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) {
    Write-Warning "CUDA PyTorch install failed; installing CPU-only PyTorch."
    & $venvPython -m pip install torch torchaudio
}

& $venvPython -m pip install -r requirements.txt

Write-Host ""
Write-Host "=== Verification ===" -ForegroundColor Cyan
ffmpeg -version | Select-Object -First 1
& $venvPython -c "import torch; print('CUDA available:', torch.cuda.is_available())"
& $venvPython -c "from faster_whisper import WhisperModel; print('faster-whisper OK')"

Write-Host ""
Write-Host "Installation complete." -ForegroundColor Green
Write-Host "Activate the environment with: .\venv\Scripts\Activate.ps1"
Write-Host "Run transcription with: python transcribe.py --help"
