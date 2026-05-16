# Variables
IMAGE_NAME=ttweekopi/student-score-prediction
# Using absolute paths for reliability across different systems
PWD=$(shell pwd)

# Volume definitions
DATA_VOL=-v $(PWD)/data:/app/data
MODEL_VOL=-v $(PWD)/data:/app/data -v $(PWD)/models:/app/models
LOG_VOL=-v $(PWD)/logs:/app/logs

# Default model variable (can be overridden with MODEL=lr in terminal)
MODEL ?= rf
# If 'model' (lowercase) is provided in the command line, override MODEL
ifdef model
    MODEL := $(model)
endif

.PHONY: build pull preprocess train evaluate all
build:
	docker build -t $(IMAGE_NAME) .

pull:
	docker pull $(IMAGE_NAME)

preprocess:
	docker run $(DATA_VOL) $(LOG_VOL) $(IMAGE_NAME) src.preprocessing

train:
	docker run $(MODEL_VOL) $(LOG_VOL) $(IMAGE_NAME) src.train --model $(MODEL)

evaluate:
	docker run $(MODEL_VOL) $(LOG_VOL) $(IMAGE_NAME) src.evaluation --model $(MODEL)

all: build preprocess train evaluate

serve: build
	python3 -m uvicorn src.serve:app --reload --port 8000

# If using Docker to serve
docker-serve:
	docker run -p 8000:8000 $(IMAGE_NAME)
	