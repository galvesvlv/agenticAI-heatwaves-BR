# AgenticAI Heatwaves BR

Heatwave prediction system for Brazil built as an end-to-end operational pipeline:

ERA5 ingestion → heatwave-condition derivation (WMO-style) → spatiotemporal deep learning inference (U-Net + temporal Transformer) → agent-based interpretation and structured report generation → Streamlit frontend for execution and visualization.

Project Example link: 
- https://drive.google.com/file/d/1Ur477gjBgrUuxl6FZcgwXHqMjR8sR9ja/view?usp=sharing
---

## 1. Project Objective

This project delivers monthly forecasts of **heatwave-condition days (HWC)** over Brazil, grounded on a WMO-style anomaly-based rule.

A heatwave-condition day is defined as:

- A day in which the daily maximum 2m air temperature exceeds the mean monthly maximum temperature by **at least 5 °C**, relative to a climatological baseline.

The system is designed to be:

- Reproducible
- Modular
- Docker-deployable
- Scientifically grounded

Deployment exposes:

- A **FastAPI backend** (`/forecast`) that executes deterministic inference and agent-based reporting.
- A **Streamlit frontend** for interactive forecast requests and visualization.

---

## 2. High-Level Architecture

### 2.1 Data + ETL Layer

- Downloads ERA5 daily maximum 2m temperature over Brazil.
- Stores data as **Zarr** for efficient incremental updates.
- Tracks dataset metadata in **MongoDB**.
- Computes monthly **HWC** (heatwave-condition day counts) using the 1961–1990 climatological reference baseline.

Outputs:
- Zarr archive (daily temperature)
- NetCDF monthly HWC dataset

---

### 2.2 Model Layer (Deep Learning)

Forecasting is performed using a **Temporal U-Net + Transformer** architecture:

- U-Net encoder-decoder:
  - Learns spatial representations.
- Transformer module at bottleneck:
  - Learns temporal dependencies across monthly sequences.

Workflow:
- Input: 12 previous months of HWC anomalies.
- Model predicts next-month anomaly field.
- Absolute HWC values reconstructed by adding month-specific climatology.

The model is fully deterministic at inference time.

---

### 2.3 Agents + RAG Layer

Uses **smolagents** to structure scientific interpretation.

Two agents are involved:

1. **Vision-Language Agent**
   - Reads the forecast map.
   - Describes spatial patterns (hotspots, gradients, contrasts).
   - Avoids hallucinated numeric claims.

2. **Orchestrator Agent**
   - Produces a structured scientific report.
   - Combines:
     - WMO definition (static curated reference).
     - Vision agent interpretation.
     - State-level structured statistics (JSON file).

RAG support includes:
- A curated WMO definition retrieval tool.
- Optional contextual web retrieval.
- Structured JSON statistics used as the quantitative backbone of the report (used just for example in this project but improves a lot in the scientific report generation by the agent).

---

### 2.4 Application Layer

**FastAPI Backend**

Endpoint:
- `/forecast`

Returns:
- Base64-encoded forecast map (PNG)
- Structured scientific report (text)

**Streamlit Frontend**

Features:
- Month selection
- Forecast trigger
- Report visualization
- Map display

The frontend communicates directly with the FastAPI container.

---

## 3. Folder Structure

```text
proj-AgenticAI-heatwaves-br/
├─ docker-compose.yml
├─ example.md
├─ readme.md
├─ etl/
|  ├─ readme.md
│  ├─ era5scraping/
│  │  ├─ src/
│  │  │  ├─ config.py
│  │  │  ├─ heatwaves_calculator.py
│  │  │  ├─ mongo_io.py
│  │  │  ├─ zarr_io.py
│  │  ├─ crontab
│  │  ├─ Dockerfile
│  │  ├─ entrypoint.sh
│  │  ├─ era5scraping.py
│  │  ├─ hw_monthly_1961-present.py
│  │  ├─ requirements.txt
│  │  ├─ run_etl.sh
│  ├─ mongodb/
│  │  ├─ Dockerfile
│  ├─ storage/
|  |  ├─ readme.md
│  │  ├─ era5_t2m_daily_max.zarr/
│  │  ├─ WMO_heatwave_conditions_1961-present.nc
├─ inference_api/
|  ├─ readme.md
│  ├─ agents/
│  │  ├─ instructions_for_orchestrator.yaml
│  │  ├─ instructions_for_vision_model.yaml
│  ├─ BR_UF_2024/
│  │  ├─ readme.md
│  │  ├─ BR_UF_2024.shp (+ sidecar files)
│  ├─ model/
│  │  ├─ readme.md
│  │  ├─ best_temporal_unet_transformer.pt
│  ├─ agents_tools.py
│  ├─ app.py
│  ├─ config.py
│  ├─ def_wmo.py
│  ├─ deterministic_tools.py
│  ├─ Dockerfile
│  ├─ final_report.py
│  ├─ model_architecture.py
│  ├─ model_preprocessing.py
│  ├─ visual_reasoning.py
│  ├─ requirements.txt
├─ web_application/
|  ├─ readme.md
│  ├─ app.py
│  ├─ Dockerfile
│  ├─ requirements.txt
└─ heatwaves-model-prediction/
   ├─ readme.md
   └─ (Google Colab notebook used for training and experiments)
```

---

## 4. Deployment

The project is containerized using Docker Compose.

Main services:

- `etl-mongo`
- `etl-scraping`
- `api`
- `frontend`

If deploying in a different environment, verify:

- API service name and port inside `docker-compose.yml`.
- The API URL configured in the Streamlit frontend.
- Required environment variables (e.g., HuggingFace token) are properly defined.

To build and start all services:

```
docker compose up --build -d
```

After startup:

- FastAPI: http://localhost:8001  
- Streamlit: http://localhost:8501  

---

## 5. Scientific and Operational Assumptions

- Baseline climatology: 1961–1990.
- Temporal resolution: monthly.
- Heatwave-condition definition: anomaly-based (+5 °C threshold).
- Severity levels: relative classification based on climatological percentiles within each state.

All outputs are **model-based forecasts**, not observations.

Results depend on:
- Training data coverage.
- Model architecture and weights.
- Normalization parameters.
- Spatial resolution and clipping behavior.

---

## 6. Research and Training Workflow

The folder:

```
heatwaves-model-prediction/
```

Contains:
- Experimental notebooks (Google Colab).
- Sequence construction logic.
- Model training experiments.
- Checkpoint generation procedures.

This folder documents the research pipeline and is not required for production inference.

---

## 7. Summary

AgenticAI Heatwaves BR integrates:

- ERA5 ingestion with incremental Zarr storage
- WMO-style heatwave-condition derivation
- Spatiotemporal deep learning inference
- Deterministic geospatial statistics
- Agent-based structured scientific reporting
- Fully Dockerized deployment with interactive frontend


The system is modular, reproducible, and designed for operational climate forecasting workflows.
