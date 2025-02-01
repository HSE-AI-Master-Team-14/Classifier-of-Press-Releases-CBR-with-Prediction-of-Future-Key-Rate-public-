FROM python:3.9-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/.local/bin
RUN curl -L http://download.cdn.yandex.net/mystem/mystem-3.1-linux-64bit.tar.gz | tar -xz -C /root/.local/bin

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt_tab

COPY best_model.joblib .
COPY best_model_preprocessor.joblib .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
