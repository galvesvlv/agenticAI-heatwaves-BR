# config.py

# Imports
from pathlib import Path

# Global Paths
OUTPUT_STORAGE_PATH = Path(__file__).resolve().parents[2] / "storage"
OUTPUT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

ZARR_PATH = OUTPUT_STORAGE_PATH / "era5_t2m_daily_max.zarr"

# MONGO VARIABLES
DATASET_NAME = "era5_t2m_brazil"
FREQ = "daily"
START_DATE = "1961-01-01"

# CDS API
DATASET = "derived-era5-single-levels-daily-statistics"
REQUEST = {
           "product_type": "reanalysis",
           "variable": ["2m_temperature"],
           "month": [
                     "01", "02", "03",
                     "04", "05", "06",
                     "07", "08", "09",
                     "10", "11", "12"
                    ],
           "day": [
                   "01", "02", "03",
                   "04", "05", "06",
                   "07", "08", "09",
                   "10", "11", "12",
                   "13", "14", "15",
                   "16", "17", "18",
                   "19", "20", "21",
                   "22", "23", "24",
                   "25", "26", "27",
                   "28", "29", "30",
                   "31"
                  ],
           "daily_statistic": "daily_maximum",
           "time_zone": "utc-03:00",
           "frequency": "1_hourly",
           "area": [
                    6,     # N
                    -75,   # E
                    -35,   # S
                    -33    # W
                    ]
          }

HISTORICAL_PERIOD = ("1961-01-01", "1990-12-31")
MONTHLY_HW_CONDITIONS_PATH = OUTPUT_STORAGE_PATH / "WMO_heatwave_conditions_1961-present.nc"