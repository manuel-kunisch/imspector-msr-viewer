# Builds dist\MSR_Viewer_v<version>\MSR_Viewer.exe (PyInstaller, one folder) and a zip of it.
# Uses its own .venv-build with the packages from requirements.txt, so only those end up in the exe.
#
#   powershell -ExecutionPolicy Bypass -File build_windows.ps1 [-SkipInstall] [-NoZip] [-Version x.y.z]

param(
    [switch]$SkipInstall,
    [switch]$NoZip,
    [string]$Version = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv-build"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$DistDir = Join-Path $ProjectRoot "dist"
$StagingDir = Join-Path $DistDir "MSR_Viewer"

Set-Location $ProjectRoot

if (-not $Version) {
    # same version as the About box: __version__ in msr_viewer.py
    $Match = Select-String -Path (Join-Path $ProjectRoot "msr_viewer.py") -Pattern '^__version__ = "(.+)"' |
        Select-Object -First 1
    if (-not $Match) {
        throw "No __version__ found in msr_viewer.py; pass -Version"
    }
    $Version = $Match.Matches[0].Groups[1].Value
}
$PackageDir = Join-Path $DistDir "MSR_Viewer_v$Version"
$ZipPath = Join-Path $DistDir "MSR_Viewer_v$Version.zip"

function Invoke-Checked {
    param(
        [Parameter(Mandatory=$true)][string]$Exe,
        [Parameter(Mandatory=$true)][string[]]$Arguments
    )

    # pip and PyInstaller log to stderr; judge them by their exit code only
    $ErrorActionPreference = "Continue"
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Compress-PackageWithRetry {
    param(
        [Parameter(Mandatory=$true)][string]$SourcePath,
        [Parameter(Mandatory=$true)][string]$DestinationPath
    )

    for ($Attempt = 1; $Attempt -le 5; $Attempt++) {
        try {
            Remove-Item -LiteralPath $DestinationPath -Force -ErrorAction SilentlyContinue
            Compress-Archive -Path $SourcePath -DestinationPath $DestinationPath -Force
            return
        } catch {
            if ($Attempt -eq 5) {
                throw
            }
            Start-Sleep -Seconds (5 * $Attempt)
        }
    }
}

if (-not (Test-Path $PythonExe)) {
    # needs Python 3.12 and the py launcher (python.org installer)
    Invoke-Checked "py" @("-3.12", "-m", "venv", $VenvDir)
}

if (-not $SkipInstall) {
    Invoke-Checked $PythonExe @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    Invoke-Checked $PythonExe @("-m", "pip", "install", "-r", "requirements.txt", "pyinstaller")
}

Invoke-Checked $PythonExe @("-m", "PyInstaller", "--noconfirm", "--clean", "msr_viewer.spec")

$ExePath = Join-Path $StagingDir "MSR_Viewer.exe"
if (-not (Test-Path $ExePath)) {
    throw "Expected executable was not created: $ExePath"
}

if (Test-Path $PackageDir) {
    Remove-Item -LiteralPath $PackageDir -Recurse -Force
}
Move-Item -LiteralPath $StagingDir -Destination $PackageDir
$ExePath = Join-Path $PackageDir "MSR_Viewer.exe"
$SizeMB = [math]::Round((Get-ChildItem -LiteralPath $PackageDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)

Write-Host "Built executable ($SizeMB MB folder):"
Write-Host $ExePath

if (-not $NoZip) {
    Compress-PackageWithRetry -SourcePath $PackageDir -DestinationPath $ZipPath
    Write-Host "Built portable zip ($([math]::Round((Get-Item $ZipPath).Length / 1MB)) MB):"
    Write-Host $ZipPath
}
