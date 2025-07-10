#!/bin/bash

# Bird Vision Environment Setup Script
# This script sets up a Python virtual environment and installs all dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect Python version
detect_python() {
    if command_exists python3.11; then
        echo "python3.11"
    elif command_exists python3.10; then
        echo "python3.10"
    elif command_exists python3.9; then
        echo "python3.9"
    elif command_exists python3; then
        echo "python3"
    elif command_exists python; then
        echo "python"
    else
        return 1
    fi
}

# Main setup function
main() {
    print_status "Setting up Bird Vision development environment..."
    
    # Check if we're in the right directory
    if [[ ! -f "pyproject.toml" ]] || [[ ! -d "src/bird_vision" ]]; then
        print_error "Please run this script from the bird_vision project root directory"
        exit 1
    fi
    
    # Detect Python
    print_status "Detecting Python installation..."
    PYTHON_CMD=$(detect_python)
    if [[ $? -ne 0 ]]; then
        print_error "Python 3.9+ not found. Please install Python 3.9 or higher."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    print_success "Found Python $PYTHON_VERSION at $($PYTHON_CMD -c 'import sys; print(sys.executable)')"
    
    # Check minimum version (3.9)
    MIN_VERSION="3.9.0"
    if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
        print_error "Python 3.9+ required, found $PYTHON_VERSION"
        exit 1
    fi
    
    # Check if virtual environment already exists
    if [[ -d "venv" ]]; then
        print_warning "Virtual environment 'venv' already exists"
        read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing existing virtual environment..."
            rm -rf venv
        else
            print_status "Using existing virtual environment..."
        fi
    fi
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        print_status "Creating virtual environment with $PYTHON_CMD..."
        $PYTHON_CMD -m venv venv
        if [[ $? -ne 0 ]]; then
            print_error "Failed to create virtual environment"
            exit 1
        fi
        print_success "Virtual environment created successfully"
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    python -m pip install --upgrade pip
    
    # Install the package in development mode
    print_status "Installing Bird Vision package in development mode..."
    
    # Ask user which installation type they want
    echo "Choose installation type:"
    echo "1) Basic installation"
    echo "2) Development installation (includes testing, linting tools)"
    echo "3) Full installation (includes ESP32 support)"
    read -p "Enter your choice (1-3) [default: 2]: " INSTALL_CHOICE
    
    case ${INSTALL_CHOICE:-2} in
        1)
            print_status "Installing basic package..."
            pip install -e .
            ;;
        2)
            print_status "Installing development package..."
            pip install -e ".[dev]"
            ;;
        3)
            print_status "Installing full package with ESP32 support..."
            pip install -e ".[dev,esp32]"
            ;;
        *)
            print_warning "Invalid choice, defaulting to development installation"
            pip install -e ".[dev]"
            ;;
    esac
    
    if [[ $? -ne 0 ]]; then
        print_error "Failed to install package dependencies"
        exit 1
    fi
    
    # Verify installation
    print_status "Verifying installation..."
    
    # Test package import
    python -c "import bird_vision; print('Package import: OK')" || {
        print_error "Package import failed"
        exit 1
    }
    
    # Test CLI availability
    if command_exists bird-vision; then
        print_success "CLI tool 'bird-vision' is available"
    else
        print_warning "CLI tool 'bird-vision' not found in PATH. You may need to restart your shell."
    fi
    
    # Show environment info
    print_status "Environment Information:"
    echo "  Python version: $(python --version)"
    echo "  Python executable: $(which python)"
    echo "  Pip version: $(pip --version)"
    echo "  Virtual environment: $(pwd)/venv"
    
    # Show next steps
    print_success "Setup completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Activate the environment: source venv/bin/activate"
    echo "2. Test the installation: bird-vision --help"
    echo "3. Run tests: python scripts/run_tests.py --unit-only"
    echo "4. See README.md for more usage instructions"
    echo
    echo "To deactivate the environment later, run: deactivate"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Bird Vision Environment Setup Script"
            echo
            echo "Usage: $0 [options]"
            echo
            echo "Options:"
            echo "  -h, --help     Show this help message"
            echo "  --force        Force recreation of virtual environment"
            echo
            echo "This script will:"
            echo "  1. Detect Python 3.9+ installation"
            echo "  2. Create a virtual environment (venv/)"
            echo "  3. Install Bird Vision package in development mode"
            echo "  4. Verify the installation"
            exit 0
            ;;
        --force)
            if [[ -d "venv" ]]; then
                print_status "Force flag specified, removing existing environment..."
                rm -rf venv
            fi
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main setup
main