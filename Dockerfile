FROM python:3.9-slim

WORKDIR /app
COPY . /app

# Install necessary packages using apt
RUN apt-get update && apt-get install -y awscli ffmpeg libsm6 libxext6 unzip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]
