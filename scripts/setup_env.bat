@echo off
REM Bird Vision Environment Setup Script for Windows
REM This script sets up a Python virtual environment and installs all dependencies

setlocal enabledelayedexpansion

REM Colors (basic Windows command prompt doesn't support colors, but we'll use echo for status)
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "WARNING=[WARNING]"
set "ERROR=[ERROR]"

REM Function to print status (simulated with goto labels)
goto main

:print_status
echo %INFO% %~1
goto :eof

:print_success
echo %SUCCESS% %~1
goto :eof

:print_warning
echo %WARNING% %~1
goto :eof

:print_error
echo %ERROR% %~1
goto :eof

:detect_python
REM Try to find Python executable
where python >nul 2>&1
if %errorlevel% == 0 (
    python --version | findstr "3\.[9-9]\|3\.1[0-9]" >nul
    if !errorlevel! == 0 (
        set "PYTHON_CMD=python"
        goto :eof
    )
)

where python3 >nul 2>&1
if %errorlevel% == 0 (
    python3 --version | findstr "3\.[9-9]\|3\.1[0-9]" >nul
    if !errorlevel! == 0 (
        set "PYTHON_CMD=python3"
        goto :eof
    )
)

REM Python not found or wrong version
set "PYTHON_CMD="
goto :eof

:main
call :print_status "Setting up Bird Vision development environment..."

REM Check if we're in the right directory
if not exist "pyproject.toml" (
    call :print_error "pyproject.toml not found. Please run this script from the bird_vision project root directory"
    exit /b 1
)

if not exist "src\bird_vision" (
    call :print_error "src\bird_vision directory not found. Please run this script from the bird_vision project root directory"
    exit /b 1
)

REM Detect Python
call :print_status "Detecting Python installation..."
call :detect_python

if "%PYTHON_CMD%"=="" (
    call :print_error "Python 3.9+ not found. Please install Python 3.9 or higher and ensure it's in your PATH."
    echo.
    echo Download Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    exit /b 1
)

REM Show Python version
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set "PYTHON_VERSION=%%i"
call :print_success "Found Python %PYTHON_VERSION%"

REM Check if virtual environment already exists
if exist "venv" (
    call :print_warning "Virtual environment 'venv' already exists"
    set /p "REPLY=Do you want to remove it and create a new one? (y/N): "
    if /i "!REPLY!"=="y" (
        call :print_status "Removing existing virtual environment..."
        rmdir /s /q venv
    ) else (
        call :print_status "Using existing virtual environment..."
    )
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    call :print_status "Creating virtual environment..."
    %PYTHON_CMD% -m venv venv
    if !errorlevel! neq 0 (
        call :print_error "Failed to create virtual environment"
        exit /b 1
    )
    call :print_success "Virtual environment created successfully"
)

REM Activate virtual environment
call :print_status "Activating virtual environment..."
call venv\Scripts\activate.bat

REM Upgrade pip
call :print_status "Upgrading pip..."
python -m pip install --upgrade pip

REM Install the package
call :print_status "Installing Bird Vision package..."
echo.
echo Choose installation type:
echo 1) Basic installation
echo 2) Development installation (includes testing, linting tools)
echo 3) Full installation (includes ESP32 support)
set /p "INSTALL_CHOICE=Enter your choice (1-3) [default: 2]: "

if "%INSTALL_CHOICE%"=="" set "INSTALL_CHOICE=2"

if "%INSTALL_CHOICE%"=="1" (
    call :print_status "Installing basic package..."
    pip install -e .
) else if "%INSTALL_CHOICE%"=="2" (
    call :print_status "Installing development package..."
    pip install -e .[dev]
) else if "%INSTALL_CHOICE%"=="3" (
    call :print_status "Installing full package with ESP32 support..."
    pip install -e .[dev,esp32]
) else (
    call :print_warning "Invalid choice, defaulting to development installation"
    pip install -e .[dev]
)

if !errorlevel! neq 0 (
    call :print_error "Failed to install package dependencies"
    exit /b 1
)

REM Verify installation
call :print_status "Verifying installation..."

python -c "import bird_vision; print('Package import: OK')"
if !errorlevel! neq 0 (
    call :print_error "Package import failed"
    exit /b 1
)

REM Test CLI availability
where bird-vision >nul 2>&1
if !errorlevel! == 0 (
    call :print_success "CLI tool 'bird-vision' is available"
) else (
    call :print_warning "CLI tool 'bird-vision' not found in PATH. You may need to restart your command prompt."
)

REM Show environment info
call :print_status "Environment Information:"
python --version
echo   Python executable: 
where python
pip --version
echo   Virtual environment: %CD%\venv

REM Show next steps
call :print_success "Setup completed successfully!"
echo.
echo Next steps:
echo 1. Activate the environment: venv\Scripts\activate.bat
echo 2. Test the installation: bird-vision --help
echo 3. Run tests: python scripts\run_tests.py --unit-only
echo 4. See README.md for more usage instructions
echo.
echo To deactivate the environment later, run: deactivate

endlocal
pause