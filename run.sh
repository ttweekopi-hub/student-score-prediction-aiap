#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define console text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}[1/5] Initializing Automated ML Pipeline...${NC}"

# ... [Any virtual environment activation steps go here if you use them] ...

echo -e "${GREEN}[2/5] Environment Ready.${NC}"

echo -e "${YELLOW}[3/5] Ingesting and Preprocessing Data...${NC}"
# Run preprocessing as a module from root
python -m src.preprocessing

echo -e "${YELLOW}[4/5] Training the Model...${NC}"
# Run training as a module from root
python -m src.train

echo -e "${YELLOW}[5/5] Evaluating Model Performance...${NC}"
# Run evaluation as a module from root
python -m src.evaluation

echo -e "${GREEN}=========================================================${NC}"
echo -e "${GREEN}Pipeline Completed Successfully! check pipeline.log${NC}"
echo -e "${GREEN}=================================================${NC}"