FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8051
EXPOSE 8051
CMD ["gunicorn","-w","2","-k","gthread","-b","0.0.0.0:${PORT}","presentation_app.app:app.server"]