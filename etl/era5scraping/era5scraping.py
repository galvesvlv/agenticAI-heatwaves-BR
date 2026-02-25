# era5scraping.py

# Imports
import cdsapi
import sys
from pathlib import Path
from datetime import datetime
import xarray as xr

# Modules
from src.config import (
                        OUTPUT_STORAGE_PATH,
                        ZARR_PATH,
                        DATASET, 
                        REQUEST,
                        DATASET_NAME,
                        FREQ,
                        START_DATE
                        )

from src.zarr_io import (
                         update_zarr, 
                         cleanup_nc
                         )

from src.mongo_io import (
                          get_collection,
                          get_dataset_meta,
                          needs_update,
                          upsert_dataset_meta
                          )

# Date
end_date = f"{datetime.now():%Y-%m-%d}"
end_year = datetime.now().year

# MongoDB info
collection = get_collection()
meta = get_dataset_meta(collection, DATASET_NAME)

# Checking start download
if ZARR_PATH.exists():
    try:
        with xr.open_zarr(ZARR_PATH) as ds_zarr:
            last_year = int(str(ds_zarr.time.max().values)[:4])
        start_year = last_year + 1
    except Exception:
        start_year = int(START_DATE[:4])
else:
    start_year = int(START_DATE[:4])

if not needs_update(meta, end_date):
    print("Zarr is up to date. Nothing to do.")
    sys.exit(0)

# Download
client = cdsapi.Client()

for year in range(start_year, end_year + 1):

    print(f"Processing year {year}...")

    # Base year - Request
    request_year = REQUEST.copy()
    request_year["year"] = [str(year)]

    nc_file = OUTPUT_STORAGE_PATH / f"era5_t2m_daily_max_{year}.nc"

    # Download
    client.retrieve(DATASET, request_year).download(nc_file)

    try:
        # Update Zarr file
        written_times = update_zarr(
                                    files=[nc_file],
                                    zarr_path=ZARR_PATH
                                    )

        # Erase NetCDF
        if written_times.size > 0:
            cleanup_nc([nc_file], written_times)  # type: ignore

        else:
            nc_file.unlink()

    except Exception:
        print(f"Processing error in year {year}. File preserved.")
        raise

# Update MongoDB metadata
with xr.open_zarr(ZARR_PATH) as ds_zarr:
    final_end_date = str(ds_zarr.time.max().values)[:10]

upsert_dataset_meta(
                    collection=collection,
                    dataset_name=DATASET_NAME,
                    path=str(ZARR_PATH),
                    start_date=START_DATE,
                    end_date=final_end_date,
                    freq=FREQ,
                    status="ok"
                    )
