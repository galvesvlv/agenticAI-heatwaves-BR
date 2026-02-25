# heatwaves_calculator.py
import xarray as xr

class HW_WMO_Calculator:
    """
    Heatwave calculator based on WMO-like threshold for max daily temperatures dataset.
    """

    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold

    def compute(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Compute monthly heatwave-condition days (HWC) from daily temperature anomalies.

        This method identifies days with heatwave conditions based on a fixed
        anomaly threshold (for WMO methodology) applied to daily maximum 2 m air temperature anomalies
        (t2m_anom). A heatwave-condition day is defined as any day for which
        t2m_anom > threshold, resulting in a binary indicator (1 = condition met,
        0 = otherwise). The daily indicators are then aggregated to monthly
        totals by summation. For WMO methodology this threshold is 5.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset containing a variable named ``t2m_anom`` with daily max
            temperature anomalies and a ``time`` coordinate. The time dimension
            must have daily resolution.

        Returns
        -------
        xarray.Dataset
            Dataset containing the variable ``HWC``, representing the monthly
            number of days with heatwave conditions. The dataset is resampled to
            monthly frequency (month start, ``1MS``), and the original
            ``t2m_anom`` variable is removed.

        Notes
        -----
        - The anomaly threshold is defined by the ``threshold`` attribute of the
          class (default by WMO: 5.0 °C).
        - This method assumes that the input dataset has already been converted
          to temperature anomalies relative to a climatological baseline.
        - If the input data are not daily, the resulting counts will not
          represent the number of days and should be interpreted with caution.
        """

        ds = ds.copy()

        ds["HWC"] = (ds["t2m_anom"] > self.threshold).astype("int8")
        ds["HWC"].attrs["description"] = "Heatwave condition days (anomaly > threshold)"

        ds = ds.drop_vars(["t2m_anom"])

        ds_monthly = ds.resample(time="1MS").sum()

        return ds_monthly
