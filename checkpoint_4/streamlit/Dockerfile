FROM python:3.9-slim-buster

WORKDIR /app

COPY streamlit/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit/ .

CMD ["streamlit", "run", "app.py"]
