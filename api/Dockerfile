# content_analyzer_project/api/Dockerfile

# Match your venv Python version
FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements_api.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY main.py .
COPY ml_logic.py .

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]