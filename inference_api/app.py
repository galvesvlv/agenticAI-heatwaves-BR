# app.py

# imports
from fastapi import FastAPI
from pydantic import BaseModel
import base64
from io import BytesIO
import yaml
import xarray as xr
import geopandas as gpd
from PIL import Image
import matplotlib.pyplot as plt
from smolagents import InferenceClientModel, CodeAgent

from inference_api.deterministic_tools import DeterministicTools
from inference_api.config import (
                                  DATASET_PATH,
                                  BRSHP_PATH,
                                  HISTORICAL_PERIOD,
                                  AGENTS_INSTRUCTIONS_1,
                                  AGENTS_INSTRUCTIONS_2,
                                  EPSG
                                  )
from inference_api.agents_tools import retrieve_WMO_definition, retrieve_contextual_references
from inference_api.visual_reasoning import visual_reasoning
from inference_api.final_report import generate_final_report

# API
app = FastAPI()

# Forecast Request
class ForecastRequest(BaseModel):
    target_time: str

# AI Agents Pipeline
dshw = xr.open_dataset(DATASET_PATH)
brazil = gpd.read_file(BRSHP_PATH).to_crs(epsg=EPSG)

# Visual Reasoning Agent 
with open(AGENTS_INSTRUCTIONS_1) as f:
    instructions1 = yaml.safe_load(f)

vision_model = InferenceClientModel(model_id="Qwen/Qwen3-VL-8B-Instruct")

agent_vision = CodeAgent(
                         model=vision_model,
                         tools=[
                                retrieve_WMO_definition,  # type: ignore
                                retrieve_contextual_references
                                ],
                         instructions=instructions1,
                         add_base_tools=False,
                         )

# Orchestrator Agent
with open(AGENTS_INSTRUCTIONS_2) as f:
    instructions2 = yaml.safe_load(f)

orchestrator_model = InferenceClientModel(model_id="meta-llama/Meta-Llama-3-8B-Instruct")

# Endpoint POST
@app.post("/forecast")
def run_forecast(request: ForecastRequest):

    target_time = request.target_time

    # Deterministic pipeline
    hw_pipeline = DeterministicTools(
                                     dshw=dshw,
                                     historical_period=HISTORICAL_PERIOD,
                                     target_date=target_time
                                     )

    # Render map to memory instead of file
    hw_pipeline.render_prediction_map(
                                      shapefile=brazil,
                                      target_date=target_time
                                      )

    # Compute stats for RAG
    hw_pipeline.compute_state_statistics(  # Need to check the number of tokens available in the free tier.
                                         shapefile=brazil,
                                         target_date=target_time
                                         )

    # Load image and encode as base64
    img_path = f"inference_api/agents/heatwave_prediction_{target_time}.png"
    img = Image.open(img_path)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Visual reasoning
    visual_text = visual_reasoning(
                                   agent=agent_vision,
                                   png_path=img_path
                                   )

    # Final report
    stats_path = f"inference_api/agents/state_statistics_{target_time}.json"

    final_output = generate_final_report(
                                         model=orchestrator_model,
                                         visual_text=visual_text,
                                         stats_path=stats_path,
                                         )

    report_text = final_output.content if hasattr(final_output, "content") else str(final_output)

    return {
            "target_time": target_time,
            "image_base64": img_base64,
            "report": report_text,
            }

