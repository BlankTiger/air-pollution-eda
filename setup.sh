## The file was generated with the aid from ChatGPT4

#!/bin/bash

# Check for Python 3
if command -v python3 &>/dev/null; then
    echo "Python 3 is installed."
    # Optionally check for a specific version of Python 3
    PYTHON3_VERSION=$(python3 --version)
    PYTHON3_VERSION=${PYTHON3_VERSION#"Python "}
    MINIMUM_VERSION="3.8"
    if [[ $(echo -e "$PYTHON3_VERSION\n$MINIMUM_VERSION" | sort -V | head -n1) == "$MINIMUM_VERSION" ]]; then
        echo "Python 3 version is $PYTHON3_VERSION, which meets the minimum requirement."
    else
        echo "Python 3 version is $PYTHON3_VERSION. A version $MINIMUM_VERSION or newer is required."
        exit 1
    fi
else
    echo "Python 3 is not installed. Please install Python 3."
    exit 1
fi

# Check if a virtual environment is currently activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "A virtual environment is currently activated."
    echo "Deactivating the current virtual environment."
    deactivate
else
    echo "No virtual environment is activated."
fi

# Check for the existence of the 'venv' directory
if [ -d "venv" ]; then
    echo "'venv' directory exists."
else
    echo "'venv' directory does not exist. Creating a virtual environment."
    # Use Python 3 to create a virtual environment named 'venv'
    python3 -m venv venv
    echo "Virtual environment 'venv' has been created."
fi

# Note: Activating a venv within a script only affects commands within the same script.
echo "Activating the virtual environment."
source venv/bin/activate

# Check if your installed packages are still up-to-date
pip install -r requirements.txt

# Open intro notebook using jupyter lab
jupyter lab notebooks/intro.ipynb
