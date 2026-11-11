$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " Satyameter - Windows setup" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

try {
    & py -3.12 --version | Out-Host
} catch {
    Write-Host "Python 3.12 was not found. Install 64-bit Python 3.12 from python.org and enable the Python Launcher." -ForegroundColor Red
    exit 1
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$recreateVenv = -not (Test-Path $python)
if (-not $recreateVenv) {
    $existingVersion = & $python --version 2>&1 | Out-String
    if ($existingVersion -notmatch "Python 3\.12\.") {
        Write-Host "Existing .venv is not Python 3.12 ($existingVersion). Recreating it..." -ForegroundColor Yellow
        $recreateVenv = $true
    }
}
if ($recreateVenv) {
    if (Test-Path ".venv") { Remove-Item ".venv" -Recurse -Force }
    Write-Host "Creating Python 3.12 virtual environment..." -ForegroundColor Yellow
    & py -3.12 -m venv .venv
}
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

Write-Host "Installing CPU-only PyTorch..." -ForegroundColor Yellow
& $python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "CPU PyTorch installation failed." }

Write-Host "Installing project dependencies..." -ForegroundColor Yellow
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed." }

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Add your Groq API key to .env, or switch to Ollama." -ForegroundColor Yellow
} else {
    Write-Host "Existing .env found; it was left untouched." -ForegroundColor Green
}

Write-Host "Running environment checks..." -ForegroundColor Yellow
& $python -m satyameter.cli doctor
if ($LASTEXITCODE -ne 0) {
    Write-Host "Some checks need attention. Read the messages above before starting the app." -ForegroundColor Yellow
} else {
    Write-Host "Setup completed. Start the app with .\run_app.bat" -ForegroundColor Green
}
