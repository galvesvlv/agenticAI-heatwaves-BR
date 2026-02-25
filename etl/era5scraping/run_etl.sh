#!/bin/bash
set -e

echo "Starting ERA5 scraping..."
python /app/etl/era5scraping.py

echo "Starting heatwave monthly processing..."
python /app/etl/hw_monthly_1961-present.py

echo "ETL finished successfully."