# config.py

# imports
from pathlib import Path
import torch

# Global Variables
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "inference_api/model/best_temporal_unet_transformer.pt"
DATASET_PATH = ROOT_DIR / "etl/storage/WMO_heatwave_conditions_1961-present.nc"
BRSHP_PATH = ROOT_DIR / "inference_api/BR_UF_2024/BR_UF_2024.shp"
HISTORICAL_PERIOD = ("1961-01-01", "1990-12-31")
AGENTS_FOLDER = ROOT_DIR / "inference_api/agents"
AGENTS_INSTRUCTIONS_1 = AGENTS_FOLDER / "instructions_for_vision_model.yaml"
AGENTS_INSTRUCTIONS_2 = AGENTS_FOLDER / "instructions_for_orchestrator.yaml"

# Target time
EPSG = 4326

# Device
DEVICE =  "cuda" if torch.cuda.is_available() else "cpu"