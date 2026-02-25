# inference_api

This folder contains the **operational inference and reporting layer** of AgenticAI Heatwaves BR.

It is responsible for:

- Loading the deterministic HWC dataset.
- Running the deep learning model for anomaly prediction.
- Reconstructing absolute heatwave-condition values.
- Rendering geospatial forecast maps.
- Computing state-level statistics.
- Executing agent-based scientific report generation.
- Exposing a FastAPI endpoint for external consumption.

This layer transforms preprocessed climate data into structured, decision-ready outputs.

---

## 1. Folder Structure

```text
inference_api/
├─ readme.md
├─ agents/
│  ├─ instructions_for_orchestrator.yaml
│  ├─ instructions_for_vision_model.yaml
|  ├─ heatwave_prediction_YYYY-MM-DD.png (generated during the communication with the frontend)
|  ├─ state_statistics_YYYY-MM-DD.json (generated during the communication with the frontend)
├─ BR_UF_2024/
│  ├─ readme.md
│  ├─ BR_UF_2024.shp (+ sidecar files)
├─ model/
|  ├─ readme.md
│  ├─ best_temporal_unet_transformer.pt
├─ agents_tools.py
├─ app.py
├─ config.py
├─ def_wmo.py
├─ deterministic_tools.py
├─ Dockerfile
├─ final_report.py
├─ model_architecture.py
├─ model_preprocessing.py
├─ visual_reasoning.py
├─ requirements.txt
```

---

## 2. High-Level Responsibilities

The inference layer performs the following sequence:

1. Load monthly HWC dataset.
2. Compute climatology and anomalies.
3. Generate anomaly forecast using the trained model.
4. Reconstruct absolute HWC values.
5. Render spatial prediction map.
6. Compute state-level statistics.
7. Use agent-based reasoning to produce structured scientific report.
8. Return results through a FastAPI endpoint.

All operations are deterministic except the LLM-based reporting stage.

---

## 3. Core Modules

### 3.1 app.py

Defines the FastAPI application.

Endpoint:

```
POST /forecast
```

Input:
```
{
  "target_time": "YYYY-MM-DD"
}
```

Workflow inside endpoint:

- Instantiate DeterministicTools.
- Render forecast map.
- Compute state-level statistics.
- Run vision-language agent.
- Run orchestrator agent.
- Return:
  - Base64 image
  - Structured scientific report
  - Target time

This file acts as the orchestration entry point.

---

### 3.2 deterministic_tools.py

Implements the full deterministic pipeline.

Main class:
- DeterministicTools

Responsibilities:

- Extract HWC variable.
- Compute monthly climatology (1961–1990).
- Compute anomalies.
- Select last 12 months.
- Normalize input.
- Load model checkpoint.
- Run inference.
- Inverse transform prediction.
- Reconstruct absolute HWC.
- Render forecast map (PNG).
- Compute state-level statistics (JSON).

---

### 3.3 model_architecture.py

Defines:

Class:
- TemporalUnetTransformer

Architecture:
- U-Net encoder-decoder (ResNet18 backbone).
- Transformer encoder at bottleneck.
- Spatial prediction of next-month anomaly field.

This file must match the architecture used during training.

---

### 3.4 model_preprocessing.py

Defines:

- ZScoreNormalizer
- DHWDataset
- make_sequences()

These are reused from the training phase to ensure:

- Consistent normalization.
- Compatible tensor shapes.
- Deterministic inference preprocessing.

---

### 3.5 final_report.py

Handles structured report generation.

Responsibilities:

- Load state-level statistics JSON.
- Inject WMO definition.
- Inject visual interpretation text.
- Construct structured prompt.
- Call LLM to generate final report.

Output structure enforced:

- Overview
- Spatial Patterns
- State-Level Highlights
- Scientific Context
- Limitations

This module isolates LLM prompting logic.

---

### 3.6 visual_reasoning.py

Runs the vision-language agent.

Input:
- Forecast PNG image.

Output:
- Qualitative spatial description.

Constraints:
- No numeric hallucination.
- No speculative causality.
- Focus on visible spatial patterns.

---

### 3.7 agents_tools.py

Defines tools available to agents.

Includes:

- retrieve_WMO_definition()
- retrieve_contextual_references(query)

These enable:

- Controlled grounding in institutional references.
- Optional contextual RAG support from web.

---

### 3.8 def_wmo.py

Contains curated WMO definition.

Purpose:

- Ensure conceptual grounding.
- Maintain consistency with operational definition.
- Avoid reliance on dynamic retrieval for core terminology.

---

### 3.9 config.py

Defines:

- Dataset path
- Model path
- Shapefile path
- Baseline period
- Agents instruction paths
- Device selection (CPU/GPU)

Centralizes configuration for reproducibility.

---

## 4. Agents Directory

Location:
```
inference_api/agents/
```

Contains:

- YAML instruction files.
- Generated PNG maps.
- Generated JSON state statistics.

This directory acts as a transient workspace for each forecast execution.

---

## 5. Geospatial Context

Folder:
```
BR_UF_2024/
```

Contains:

- Brazilian state boundaries shapefile.
- Used for:
  - Map overlay
  - Spatial clipping
  - State-level statistics computation

All operations assume EPSG:4326.

---

## 6. Model Checkpoint

Location:
```
inference_api/model/best_temporal_unet_transformer.pt
```

Contents:

- model_state
- mu
- sigma

Loaded at runtime for deterministic anomaly forecasting.

---

## 7. Docker Configuration

Dockerfile:

- Base image: Python 3.12
- Installs geospatial dependencies (GDAL, etc.)
- Installs Python requirements
- Exposes port 8001
- Runs uvicorn

Service name in docker-compose must match frontend configuration.

---

## 8. Execution Characteristics

Deterministic components:
- Data extraction
- Climatology computation
- Anomaly derivation
- Model inference
- Map rendering
- State statistics

Non-deterministic component:
- LLM-based report generation (temperature-controlled).

All forecasts are:

- Model-based
- Baseline-referenced (1961–1990)
- Monthly resolution
- Spatially explicit

---

## 9. Design Principles

- Clear separation between deterministic computation and agent reasoning.
- Strict grounding of heatwave definition.
- Structured JSON as quantitative backbone.
- Reproducible inference pipeline.
- Modular architecture.

---

## 10. Summary

The inference_api layer operationalizes the full forecasting system.

It bridges:

- Deterministic climate modeling
- Deep learning anomaly forecasting
- Geospatial processing
- Retrieval-augmented reasoning
- Structured scientific reporting
- Web API deployment

This folder is the production core of AgenticAI Heatwaves BR.