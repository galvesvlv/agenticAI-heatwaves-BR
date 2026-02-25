#!/bin/bash
set -e

sed -i 's/\r$//' run_etl.sh || true
sed -i 's/\r$//' crontab || true

crontab crontab
cron -f
