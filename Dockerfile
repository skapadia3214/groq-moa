FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
COPY . /app/

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port", "8080"]
