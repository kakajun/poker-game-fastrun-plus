# Check if conda is installed
if (!(Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Conda is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

# Create environment
Write-Host "Creating Conda environment 'poker-rl'..." -ForegroundColor Cyan
conda env create -f environment.yml

# Activate instructions
Write-Host "Environment created successfully!" -ForegroundColor Green
Write-Host "To activate, run: conda activate poker-rl" -ForegroundColor Yellow
