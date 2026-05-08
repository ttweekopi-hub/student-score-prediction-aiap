# Use a slim version of Python
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (src, data, config.json)
COPY . .

# Create necessary directories
RUN mkdir -p data models

# Set entrypoint to run python modules
ENTRYPOINT ["python", "-u", "-m"]

# Expose FastAPI port
EXPOSE 8000

# Default command can stay as the pipeline, 
# or you can change it to start the API
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
