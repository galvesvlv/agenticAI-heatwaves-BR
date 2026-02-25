# hw_monthly_1961-present.py

# imports
import xarray as xr
from src.config import (
                        ZARR_PATH,
                        HISTORICAL_PERIOD as HP,
                        MONTHLY_HW_CONDITIONS_PATH
                        )
from src.heatwaves_calculator import HW_WMO_Calculator

def main():
    # loading historical temperature file
    ds = xr.open_zarr(ZARR_PATH)

    # anomalies
    clim = ds["t2m"].sel(time=slice(HP[0], HP[1])).groupby('time.month').mean(dim='time')
    t2m_anom = ds['t2m'].groupby('time.month') - clim
    ds_anom = t2m_anom.to_dataset(name="t2m_anom")

    # heatwaves conditions calculation through WMO methodology
    calculation_hw = HW_WMO_Calculator()
    hw_wmo_conditions = calculation_hw.compute(ds_anom)
    hw_wmo_conditions["HWC"].attrs.update(
                                        {
                                        "long_name": "Monthly number of heatwave-condition days for WMO methodology",
                                        "threshold": "t2m max daily anomalies > 5°C",
                                        "baseline": "1961–1990"
                                        }
                                        )

    # Saving hw_wmo_conditions
    print("Starting storage of HWC for each month...")
    hw_wmo_conditions.to_netcdf(path=MONTHLY_HW_CONDITIONS_PATH)
    print("Number of days with heatwave conditions in the month ✅")

if __name__ == "__main__":
    main()
