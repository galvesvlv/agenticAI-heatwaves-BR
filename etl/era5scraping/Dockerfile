# Dockerfile ETL

FROM python:3.12-slim

WORKDIR /app/etl/era5scraping

RUN apt-get update && apt-get install -y \
    bash \
    cron \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
