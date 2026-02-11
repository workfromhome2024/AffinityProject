FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir django==6.0.2 djangorestframework==3.16.1 \
    django-cors-headers celery redis pillow

COPY affinity/ affinity/
COPY smolvla/ smolvla/
COPY manage.py .

RUN mkdir -p /app/shared_media

EXPOSE 8000

CMD python manage.py makemigrations smolvla && python manage.py migrate && python manage.py runserver 0.0.0.0:8000
