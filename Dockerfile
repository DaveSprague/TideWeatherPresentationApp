FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD gunicorn -w 2 -k gthread --threads 4 --timeout 120 -b 0.0.0.0:$PORT app:server