#!/bin/bash

# Name of the virtual environment
VENV_DIR="venv"

# Check if the virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "⚙️ Creating the virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "✅ Virtual environment already exists."
fi

# Activate the virtual environment
echo "🔄 Activating the virtual environment..."
source "$VENV_DIR/bin/activate"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: The file requirements.txt does not exist!"
    deactivate
    exit 1
fi

# Check for already installed dependencies
echo "🔍 Checking installed dependencies..."
MISSING_PACKAGES=false

while IFS= read -r package; do
    pkg_name=$(echo "$package" | cut -d= -f1)  # Extract the package name
    if ! pip show "$pkg_name" &> /dev/null; then
        echo "❗ The package $pkg_name is not installed."
        MISSING_PACKAGES=true
    fi
done < requirements.txt

# If there are missing packages, install them
if [ "$MISSING_PACKAGES" = true ]; then
    echo "📦 Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "✅ All dependencies are already installed!"
fi

echo "✅ Setup complete! The virtual environment is active."

# Keep the virtual environment active in the terminal
