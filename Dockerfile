FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# 먼저 requirements 설치
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# LightFM 은 pip로도 충분하지만 conda 없이 설치하도록
RUN pip install lightfm

RUN pip install pytest
ENV PYTHONPATH="/app"

# 소스 복사
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]