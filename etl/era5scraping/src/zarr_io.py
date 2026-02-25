# zarr_io.py

# Imports
import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Sequence

# Functions

def update_zarr(files: List[Path], zarr_path: Path) -> np.ndarray:
    """
    Incrementally updates a Zarr dataset by appending new time steps
    from a list of NetCDF files.

    If the Zarr store already exists, only time coordinates that are
    not yet present in the Zarr dataset are appended. If the Zarr store
    does not exist, it is created from scratch.

    Parameters
    ----------
    files : list of pathlib.Path
        List of NetCDF files containing new data to be ingested.
        All files must share the same variables, grid, and dimensions.
    zarr_path : pathlib.Path
        Path to the target Zarr store. The store will be created if it
        does not exist, or updated if it already exists.

    Returns
    -------
    numpy.ndarray
        Array of datetime64 values corresponding to the time coordinates
        that were successfully written to the Zarr store. If no new time
        steps are appended, an empty array is returned.
    """

    zarr_path = Path(zarr_path)

    with xr.open_mfdataset(
                           files, 
                           combine="by_coords", 
                           parallel=False
                           ) as ds_new:
        
        ds_new = ds_new.rename({k: "time" for k in ds_new.dims if k == "valid_time"})
        ds_new = ds_new.rename({k: "time" for k in ds_new.coords if k == "valid_time"})

        if zarr_path.exists():
            with xr.open_zarr(zarr_path) as ds_old:
                ds_new = ds_new.sel(time=~ds_new.time.isin(ds_old.time))

            if ds_new.time.size > 0:
                ds_new = ds_new.sortby("time")
                for var in ds_new.data_vars:
                    ds_new[var].encoding.pop("chunks", None)
                
                ds_new = ds_new.chunk(
                                      {
                                       "time": -1,
                                       "latitude": 50,
                                       "longitude": 50
                                       }
                                       )
                ds_new.to_zarr(
                               zarr_path,
                               mode="a",
                               append_dim="time"
                               )
                return ds_new.time.values

            return np.array([], dtype="datetime64[ns]")

        else:
            ds_new = ds_new.sortby("time")
            for var in ds_new.data_vars:
                    ds_new[var].encoding.pop("chunks", None)
            
            ds_new = ds_new.chunk(
                                      {
                                       "time": -1,
                                       "latitude": 50,
                                       "longitude": 50
                                       }
                                       )
            
            ds_new.to_zarr(zarr_path, mode="w")
            return ds_new.time.values


def cleanup_nc(files: List[Path], used_times: Sequence[np.datetime64]) -> None:
    """
    Removes NetCDF files whose entire time range has already been
    ingested into the target Zarr dataset.

    A file is deleted only if all of its time coordinates are present
    in `used_times`. This prevents accidental deletion of files that
    still contain unprocessed data.

    Parameters
    ----------
    files : list of pathlib.Path
        List of NetCDF files to be evaluated for removal.
    used_times : sequence of numpy.datetime64
    All time coordinates currently present in the Zarr store.

    Returns
    -------
    None
        Files are removed from disk as a side effect.
    """

    used_times = set(used_times) # type: ignore

    for f in files:
        with xr.open_dataset(f) as ds:
            ds = ds.rename({k: "time" for k in ds.dims if k == "valid_time"})
            ds = ds.rename({k: "time" for k in ds.coords if k == "valid_time"})
            file_times = set(ds.time.values)

        # Remove only if ALL times in the file were already used
        if file_times.issubset(used_times):
            f.unlink()

