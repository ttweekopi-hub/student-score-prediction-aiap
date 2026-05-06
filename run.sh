#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define console text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}    U.A. Secondary School Score Prediction Pipeline   ${NC}"
echo -e "${BLUE}======================================================${NC}"

# 1. Check if score.db is in the data folder
if [ ! -f "data/score.db" ]; then
    echo -e "${YELLOW}⚠️  Warning: Raw database 'data/score.db' not found.${NC}"
    echo -e "${BLUE}Downloading score.db automatically...${NC}"
    mkdir -p data
    curl -L -o data/score.db "https://techassessment.blob.core.windows.net/aiap-preparatory-bootcamp/score.db"
    echo -e "${GREEN}✓ Download complete.${NC}"
fi

# 2. Setup virtual environment
echo -e "\n${BLUE}[1/5] Setting up Virtual Environment (venv)...${NC}"

# Check if 'python' or 'python3' is the active command
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${YELLOW}❌ Error: Python is not installed or not added to your system PATH.${NC}"
    exit 1
fi

echo -e "Using command: ${PYTHON_CMD}"

# Create the virtual environment if it doesn't existe
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
fi

# Cross-platform activation logic (detects Windows vs Mac/Linux)
if [ -f "venv/Scripts/activate" ]; then
    # Windows paths (even inside Git Bash)
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    # macOS / Linux paths
    source venv/bin/activate
else
    echo -e "${YELLOW}❌ Error: Active script not found in venv.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Virtual environment activated.${NC}"

# 3. Install packages
echo -e "\n${BLUE}[2/5] Installing Dependencies from requirements.txt...${NC}"

# Safely upgrade pip using python module syntax to prevent file-locking on Windows
$PYTHON_CMD -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed successfully.${NC}"

# 4. Preprocess Data
echo -e "\n${BLUE}[3/5] Ingesting and Preprocessing Data...${NC}"
python src/preprocessing.py
echo -e "${GREEN}✓ Preprocessing complete. Cleaned file generated in data/cleaned_score.csv.${NC}"

# 5. Train Default Model
echo -e "\n${BLUE}[4/5] Training Best Optimized Random Forest Model...${NC}"
python src/train.py
echo -e "${GREEN}✓ Model training complete. Serialized model saved to models/student_model.pkl.${NC}"

# 6. Evaluate Model
echo -e "\n${BLUE}[5/5] Evaluating Best Model on Unseen Test Split...${NC}"
python src/evaluation.py

echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}       Pipeline Executed Successfully!                 ${NC}"
echo -e "${GREEN}======================================================${NC}"