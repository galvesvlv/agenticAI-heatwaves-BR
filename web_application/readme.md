# web_application

This folder contains the **Streamlit frontend interface** of AgenticAI Heatwaves BR.

It provides a lightweight web application that allows users to:

- Select a forecast month.
- Trigger the inference pipeline.
- Visualize the forecast map.
- Read the structured scientific report.

The frontend communicates directly with the FastAPI backend defined in the `inference_api` service.

---

## 1. Folder Structure

```text
web_application/
├─ readme.md
├─ app.py
├─ Dockerfile
├─ requirements.txt
```

---

## 2. Application Overview

The frontend is implemented using **Streamlit** and is intentionally minimal.

Its responsibilities are limited to:

- User input collection.
- Sending POST requests to the API.
- Rendering returned outputs.

All forecasting logic remains inside the backend.

---

## 3. app.py

This file defines the Streamlit application.

Core workflow:

1. Render page title.
2. Display date selector for forecast month.
3. Wait for user to click "Run Forecast".
4. Send POST request to:

```
http://api:8001/forecast
```

Request body format:

```
{
  "target_time": "YYYY-MM-DD"
}
```

5. Receive response containing:
   - target_time
   - image_base64
   - report

6. Decode base64 PNG.
7. Display:
   - Scientific report text.
   - Prediction map.

The frontend does not perform any model computation.

---

## 4. API Communication

The API URL is defined as:

```
API_URL = "http://api:8001/forecast"
```

Important considerations:

- The service name `api` must match the Docker Compose configuration.
- If renamed, this URL must be updated accordingly.
- Both frontend and API must share the same Docker network.

---

## 5. Docker Configuration

The Dockerfile:

- Uses Python 3.12-slim.
- Installs Streamlit and required dependencies.
- Exposes port 8501.
- Launches the app via:

```
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

The frontend container depends on the API container.

---

## 6. User Interaction Flow

Step-by-step:

1. User selects forecast month.
2. User clicks "Run Forecast".
3. Frontend sends request to API.
4. API executes:
   - Deterministic pipeline
   - Model inference
   - Agent-based reporting
5. API returns map + report.
6. Frontend renders results.

The UI is intentionally simple to emphasize scientific output and an example frontend interface.

---

## 7. Error Handling

If the API request fails:

- A generic error message is displayed.
- No internal stack traces are exposed.

This prevents leaking backend details.

---

## 8. Design Principles

- Separation of concerns:
  - Frontend handles UI only.
  - Backend handles computation and reasoning.
- Stateless interface.
- Minimal logic inside UI layer.
- Containerized for reproducibility.

---

## 9. Deployment Notes

After running:

```
docker compose up --build -d
```

Access the frontend at:

http://localhost:8501

Ensure that:

- The API service is running.
- Ports 8001 and 8501 are not blocked.
- Docker networking is correctly configured.

---

## 10. Summary

The web_application layer provides a minimal operational interface for:

- Triggering model-based heatwave forecasts.
- Viewing structured scientific reports.
- Visualizing spatial prediction outputs.

It acts as the human-facing component of AgenticAI Heatwaves BR while keeping all scientific and computational logic in the backend.