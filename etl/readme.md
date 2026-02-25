# ETL Layer – AgenticAI Heatwaves BR

This folder contains the full data ingestion and preprocessing pipeline responsible for:

- Downloading ERA5 daily maximum temperature data.
- Incrementally updating a Zarr archive.
- Computing monthly heatwave-condition (HWC) datasets.
- Tracking dataset metadata in MongoDB.
- Automating execution through a containerized cron service.

This layer produces the deterministic dataset consumed by the inference API.

---

## 1. ETL Folder Structure

```text
etl/
├─ readme.md
├─ era5scraping/
│  ├─ src/
│  │  ├─ config.py
│  │  ├─ heatwaves_calculator.py
│  │  ├─ mongo_io.py
│  │  ├─ zarr_io.py
│  ├─ crontab
│  ├─ Dockerfile
│  ├─ entrypoint.sh
│  ├─ era5scraping.py
│  ├─ hw_monthly_1961-present.py
│  ├─ requirements.txt
│  ├─ run_etl.sh
├─ mongodb/
│  ├─ Dockerfile
├─ storage/
|  ├─ readme.md
│  ├─ era5_t2m_daily_max.zarr/
│  ├─ WMO_heatwave_conditions_1961-present.nc
```

---

## 2. Workflow

The ETL pipeline executes the following steps:

1. Download ERA5 daily maximum 2m temperature data (year by year).
2. Append new data into a Zarr archive (incrementally).
3. Update MongoDB metadata.
4. Compute daily anomalies relative to 1961–1990.
5. Derive monthly heatwave-condition day counts (HWC).
6. Save monthly HWC dataset to NetCDF.
7. Run automatically via cron (containerized).

The result is a continuously updated monthly HWC dataset used by the inference API.

---

## 3. Directory Breakdown

### 3.1 era5scraping/

This folder contains the core ETL logic.

---

### 3.1.1 src/

Contains modularized utilities.

#### config.py

Defines:

- Output storage paths.
- Zarr target path.
- MongoDB dataset identifiers.
- ERA5 request parameters (CDS API).
- Historical climatological period (1961–1990).
- Output NetCDF path for monthly HWC.

Centralizes configuration to ensure reproducibility.

---

#### heatwaves_calculator.py

Implements:

Class: HW_WMO_Calculator

Responsibilities:
- Receives daily anomaly dataset.
- Applies anomaly threshold (default 5 °C).
- Converts to binary heatwave-condition indicator.
- Aggregates to monthly counts.

Output:
- Monthly dataset containing HWC (heatwave-condition day counts).

This module isolates the WMO-style rule implementation.

---

#### mongo_io.py

Handles metadata tracking.

Functions:

- get_collection(): connects to MongoDB.
- get_dataset_meta(): retrieves metadata record.
- needs_update(): determines if update is required.
- upsert_dataset_meta(): inserts or updates dataset metadata.

Metadata fields include:
- dataset name
- path
- start date
- end date
- frequency
- last update
- processing status

Prevents redundant ingestion and ensures state consistency.

---

#### zarr_io.py

Handles incremental Zarr operations.

Functions:

- update_zarr():
  - Opens NetCDF files.
  - Renames valid_time → time if necessary.
  - Detects new timestamps.
  - Appends only unseen time steps.
  - Rechunks data.
  - Returns written timestamps.

- cleanup_nc():
  - Deletes NetCDF files only if fully ingested.
  - Prevents accidental data loss.

This ensures safe, incremental storage.

---

### 3.1.2 era5scraping.py

Main ingestion script.

Responsibilities:

- Determines start year based on:
  - Existing Zarr archive.
  - MongoDB metadata.
- Iterates over missing years.
- Downloads ERA5 NetCDF files via CDS API.
- Calls update_zarr().
- Cleans up processed NetCDF files.
- Updates MongoDB metadata.

This script is the ingestion engine.

---

### 3.1.3 hw_monthly_1961-present.py

Monthly post-processing script.

Steps:

1. Opens Zarr dataset.
2. Computes monthly climatology (1961–1990).
3. Computes daily anomalies.
4. Applies WMO-style threshold rule.
5. Aggregates to monthly HWC.
6. Saves NetCDF:

```
etl/storage/WMO_heatwave_conditions_1961-present.nc
```

This file is the deterministic input for forecasting.

---

### 3.1.4 run_etl.sh

Shell orchestrator.

Executes:

1. era5scraping.py
2. hw_monthly_1961-present.py

Ensures ingestion and derivation run sequentially.

---

### 3.1.5 crontab

Defines automated schedule.

Default configuration:
- Runs on day 5 of each month at 00:00.

Triggers run_etl.sh inside container.

---

### 3.1.6 entrypoint.sh

Container startup script.

Responsibilities:

- Normalizes line endings.
- Registers cron job.
- Starts cron in foreground.

---

### 3.1.7 Dockerfile (era5scraping)

Defines ETL container environment.

Installs:
- Python 3.12
- Bash
- Cron
- Required Python dependencies

Ensures isolated and reproducible execution.

---

### 3.1.8 requirements.txt

Lists Python dependencies required for ETL pipeline

---

## 4. mongodb/

Contains Dockerfile for MongoDB service.

Purpose:
- Stores dataset metadata.
- Tracks ingestion status.
- Enables incremental updates.

Data volume:
- Persisted via Docker volume (mongo_data).

---

## 5. storage/

This directory stores generated datasets.

### 5.1 era5_t2m_daily_max.zarr/

- Incremental Zarr archive.
- Contains daily maximum 2m temperature.
- Structured for efficient chunked access.

### 5.2 WMO_heatwave_conditions_1961-present.nc

- Monthly heatwave-condition day counts.
- Baseline: 1961–1990.
- Used directly by inference API.

---

## 6. Execution Model

ETL can be executed in two ways:

### 6.1 Automatic (Production)

Via Docker Compose + cron:

```
docker compose up --build -d
```

Cron handles monthly updates.

---

### 6.2 Manual (Development)

Inside ETL container:

```
python era5scraping.py
python hw_monthly_1961-present.py
```

---

## 7. Design Principles

- Incremental ingestion.
- Metadata-driven updates.
- Baseline-consistent anomaly derivation.
- Deterministic reproducibility.
- Containerized automation.
- Separation of ingestion and derivation logic.

---

## 8. Output Guarantees

After successful execution:

- Zarr archive is up-to-date.
- MongoDB metadata reflects correct date range.
- Monthly HWC NetCDF is regenerated.
- Dataset is ready for inference layer consumption.

This ETL layer provides the deterministic foundation for the forecasting and agent-based reporting system.