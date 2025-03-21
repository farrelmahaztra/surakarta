FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir .

COPY . /app/

WORKDIR /app/web/backend

RUN pip install --no-cache-dir gunicorn psycopg2-binary dj-database-url

RUN mkdir -p /app/staticfiles
RUN mkdir -p /app/media

CMD ["sh", "-c", "python manage.py collectstatic --noinput && \
                    python manage.py migrate && \
                    gunicorn surakarta.wsgi:application --bind 0.0.0.0:8000"]