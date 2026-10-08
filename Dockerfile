FROM python:3.14-slim

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app/ ./app/
COPY models/ ./models/
COPY data/PS2.txt data/FS1.txt ./data/

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]