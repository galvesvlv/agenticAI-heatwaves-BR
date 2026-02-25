# deterministic_tools.py

#imports
import os
import json
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from inference_api.config import DEVICE, MODEL_PATH, AGENTS_FOLDER
from inference_api.model_architecture import TemporalUnetTransformer
from inference_api.model_preprocessing import ZScoreNormalizer

class DeterministicTools:
    """
    Deterministic pipeline for heatwave condition (HWC) analysis, prediction,
    visualization, and regional statistics.

    This class implements a fully eager, end-to-end deterministic workflow that,
    upon instantiation, performs:

    1. Extraction of the heatwave condition variable ('HWC') from an xarray Dataset.
    2. Computation of a monthly climatology over a user-defined historical period.
    3. Derivation of monthly anomalies relative to the climatology.
    4. Deep-learning-based prediction of future heatwave anomalies using a pretrained
    temporal neural network.
    5. Reconstruction of absolute heatwave condition values by adding the appropriate
    monthly climatology to the predicted anomalies.
    6. Optional spatial visualization of the predicted heatwave conditions.
    7. Optional computation of state-level summary statistics based on polygonal
    administrative boundaries.

    The class is intentionally designed to be fully eager: all intermediate products
    and the final prediction are computed during object initialization. This design
    choice ensures reproducibility and guarantees that a fully resolved prediction
    pipeline is available immediately after instantiation.

    Intended use cases include climate diagnostics, hazard assessment, and automated
    agent-based reporting pipelines for heatwave-related impacts.
    """

    def __init__(
                 self,
                 dshw: xr.Dataset,
                 historical_period: tuple,
                 target_date: str,
                 seq_len: int=12,
                 ):
        """
        Initialize and execute the complete deterministic heatwave prediction pipeline.

        This initializer immediately executes all steps of the workflow, including
        climatology computation, anomaly derivation, neural-network-based anomaly
        prediction, and reconstruction of absolute heatwave condition values.

        Parameters
        ----------
        dshw : xr.Dataset
            Input dataset containing the heatwave condition variable 'HWC' with
            dimensions (time, latitude, longitude). The time coordinate must be
            convertible to datetime and support monthly grouping.

        historical_period : tuple
            Two-element tuple defining the start and end dates of the reference
            climatological period (e.g., ("1961-01-01", "1990-12-31")).

        target_date : str
            Target date for prediction, provided as a string convertible to
            numpy.datetime64. Prediction is performed at monthly resolution.

        seq_len : int, optional
            Length of the temporal input sequence (number of past months) used as input
            to the deep learning model. Default is 12.

        Attributes Created
        ------------------
        dshw : xr.Dataset
            Original input dataset.

        historical_period : tuple
            Reference period used for climatology computation.

        target_date : str
            Target prediction date.

        seq_len : int
            Temporal sequence length used for inference.

        dhw : xr.DataArray
            Heatwave condition data extracted from the input dataset.

        clim : xr.DataArray
            Monthly climatology of heatwave conditions computed over the reference period.

        dhw_anom : xr.DataArray
            Monthly heatwave anomalies relative to the climatology.

        da_pred : xr.DataArray
            Predicted heatwave anomalies for the target month.

        dhw_prediction : xr.DataArray
            Reconstructed absolute heatwave condition values for the target month.
        """


        self.dshw = dshw
        self.historical_period = historical_period
        self.target_date = target_date
        self.seq_len = seq_len
        self.dhw = self.calculate_dataarray()
        self.clim = self.calculate_clim()
        self.dhw_anom = self.calculate_dataarray_anomalies()
        self.da_pred = self.prediction_of_hw_anomalies()
        self.dhw_prediction = self.calculate_anomalies_to_values()

    def calculate_dataarray(self):
        """
        Extract the heatwave condition DataArray from the input dataset.

        Returns
        -------
        xr.DataArray
            Heatwave condition variable ('HWC') with dimensions
            (time, latitude, longitude).

        Raises
        ------
        KeyError
            If the input dataset does not contain the variable 'HWC'.
        """

        if "HWC" not in self.dshw:
            raise KeyError("Input dataset must contain variable 'HWC'.")

        # Dataset -> DataArray
        dhw = self.dshw["HWC"]

        return dhw

    def calculate_clim(self):
        """
        Compute the monthly climatology of heatwave conditions.

        The climatology is calculated as the mean heatwave condition for each calendar
        month over the specified historical reference period.

        Returns
        -------
        xr.DataArray
            Monthly climatology indexed by month (1–12) with spatial dimensions
            (latitude, longitude).
        """

        # Monthly climatology (1961–1990)
        clim = (
                self.dhw
                .sel(time=slice(self.historical_period[0], self.historical_period[1]))
                .groupby("time.month")
                .mean("time")
                )

        return clim

    def calculate_dataarray_anomalies(self):
        """
        Compute monthly heatwave condition anomalies.

        Anomalies are defined as deviations from the monthly climatology computed
        over the reference historical period.

        Returns
        -------
        xr.DataArray
            Monthly heatwave anomalies with dimensions
            (time, latitude, longitude) and appropriate metadata describing
            the reference period.
        """

        # Monthly anomalies
        dhw_anom = self.dhw.groupby("time.month") - self.clim

        dhw_anom.name = "HWC_anomaly"
        dhw_anom.attrs["reference_period"] = (f"{self.historical_period[0]}–{self.historical_period[1]}")
        dhw_anom.attrs["description"] = "Monthly anomalies of heatwave conditions"

        return dhw_anom

    def prediction_of_hw_anomalies(self):
        """
        Predict future heatwave anomalies using a deep learning temporal model.

        This method:
        - Selects a fixed-length sequence of past monthly anomalies prior to the
        target month.
        - Normalizes the input using stored normalization statistics.
        - Performs inference using a pretrained Temporal U-Net Transformer model.
        - Inversely transforms the prediction back to physical anomaly units.

        Returns
        -------
        xr.DataArray
            Predicted heatwave anomalies for the target month with dimensions
            (time=1, latitude, longitude).

        Raises
        ------
        ValueError
            If insufficient historical data is available to build the required
            temporal input sequence.
        """

        target_month = np.datetime64(self.target_date, "M")  # month precision

        # Ensure sorted time axis
        self.dhw_anom = self.dhw_anom.sortby("time")

        # Select all months strictly before the target month
        hist = self.dhw_anom.sel(time=slice(None, target_month))  # includes <= target_month if present
        hist = hist.sel(time=hist.time < target_month)       # strictly before

        n_hist = hist.sizes["time"]
        if n_hist < self.seq_len:
            first = str(self.dhw_anom.time.values[0])
            last = str(self.dhw_anom.time.values[-1])
            raise ValueError(
                             f"Not enough history to predict {str(target_month)}. "
                             f"Need {self.seq_len} months strictly before target, found {n_hist}. "
                             f"Available data range: {first} .. {last}."
                             )

        # Take the last seq_len months
        window = hist.isel(time=slice(-self.seq_len, None))

        # Tensor preparation: (B=1, T, C=1, H, W)
        X = window.transpose("time", "latitude", "longitude").values
        X = X[None, :, None, :, :]  # (1, T, 1, H, W)
        X = torch.from_numpy(X).float().to(DEVICE)

        # Load model and normalizer checkpoint
        model = TemporalUnetTransformer().to(DEVICE)
        normalizer = ZScoreNormalizer()

        ckpt = torch.load(str(MODEL_PATH), map_location=DEVICE)

        model.load_state_dict(ckpt["model_state"])
        normalizer.mu = ckpt["mu"].to(DEVICE)
        normalizer.sigma = ckpt["sigma"].to(DEVICE)

        model.eval()

        # Normalize input -> predict -> inverse transform
        X_norm = normalizer.transform(X)

        with torch.no_grad():
            y_hat_norm = model(X_norm)

        y_hat_anom = normalizer.inverse_transform(y_hat_norm)  # (1, 1, H, W)

        # Back to xarray (time dimension length 1 at target_month)
        y_map = y_hat_anom[0, 0].cpu().numpy()

        da_pred = xr.DataArray(
                               y_map[None, :, :],
                               dims=("time", "latitude", "longitude"),
                               coords={
                                       "time": [target_month.astype("datetime64[ns]")],
                                       "latitude": self.dhw_anom.latitude,
                                       "longitude": self.dhw_anom.longitude,
                                       },
                               name="HWC_anomaly_pred",
                               )

        da_pred.attrs["reference_time"] = str(target_month)
        da_pred.attrs["sequence_length"] = self.seq_len
        da_pred.attrs["model_checkpoint"] = "best_temporal_unet_transformer.pt"

        return da_pred

    def calculate_anomalies_to_values(self):
        """
        Reconstruct absolute heatwave condition values from predicted anomalies.

        The reconstruction is performed by adding the predicted anomaly field to
        the monthly climatology corresponding to the target prediction month.

        Returns
        -------
        xr.DataArray
            Predicted absolute heatwave condition values with dimensions
            (time=1, latitude, longitude).
        """

        # Extract month from reference time
        ref_time = np.datetime64(self.da_pred.attrs["reference_time"])
        month = int(np.datetime_as_string(ref_time, unit="M").split("-")[1])

        # Add climatology
        dhw_prediction = self.da_pred + self.clim.sel(month=month)

        dhw_prediction.name = "HWC_prediction"
        dhw_prediction.attrs["reference_time"] = self.da_pred.attrs["reference_time"]

        return dhw_prediction

    def render_prediction_map(
                              self,
                              shapefile: gpd.GeoDataFrame,
                              target_date: str,
                              cmap_name: str="Reds",
                              bounds: np.linspace=None,
                              title_prefix: str="Predicted Heatwave Condition Days",
                              colorbar_label: str="Number of heatwave days",
                              filename_prefix: str="heatwave_prediction"
                              ):
        """
        Render and save a spatial map of predicted heatwave conditions.

        This method generates a cartographic visualization of the predicted heatwave
        condition values, overlays administrative boundaries from a provided shapefile,
        and saves the resulting figure to disk.

        Parameters
        ----------
        shapefile : geopandas.GeoDataFrame
            GeoDataFrame containing polygon geometries (e.g., country or state borders)
            used for spatial context and visualization.

        target_date : str
            Target date associated with the prediction, used for labeling and output
            file naming.

        cmap_name : str, optional
            Name of the Matplotlib colormap used for rendering the prediction.
            Default is "Reds".

        bounds : np.ndarray, optional
            Explicit colorbar boundaries. If None, bounds are computed automatically
            based on the data distribution (0 to 95th percentile).

        title_prefix : str, optional
            Prefix used in the figure title. Default is "Predicted Heatwave Conditions".

        colorbar_label : str, optional
            Label for the colorbar. Default is "Number of heatwave days".

        filename_prefix : str, optional
            Prefix used when saving the output image file. Default is
            "heatwave_prediction".

        Raises
        ------
        ValueError
            If the predicted DataArray does not contain latitude and longitude
            dimensions.

        Notes
        -----
        The output image is saved to the configured agents output directory in PNG
        format with high resolution (300 dpi).
        """

        cmap = plt.get_cmap(cmap_name)

        da = self.dhw_prediction.copy()

        if not {"latitude", "longitude"}.issubset(da.dims):
            raise ValueError(
                             "DataArray must have 'latitude' and 'longitude' dimensions."
                             )

        if not da.rio.crs:
            da = da.rio.write_crs(4326)

        if da.latitude[0] > da.latitude[-1]:
            da = da.sortby("latitude")

        if "time" in da.dims:
            da = da.isel(time=0)

        os.makedirs(AGENTS_FOLDER, exist_ok=True)

        if bounds is None:
            data = da.values
            data = data[np.isfinite(data)]
            vmin = max(0.0, float(data.min()))
            vmax = round(np.percentile(data, 95), 0)
            bounds = np.linspace(vmin, vmax, 11)

        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        lon, lat = np.meshgrid(
                               da.longitude.values,
                               da.latitude.values
                               )

        fig, ax = plt.subplots(
                               figsize=(10, 10),
                               subplot_kw={"projection": ccrs.PlateCarree()}
                               )

        shapefile.plot(
                       ax=ax,
                       facecolor="none",
                       edgecolor="black",
                       linewidth=0.8,
                       zorder=10,
                       )

        ax.scatter(
                   lon,
                   lat,
                   c=da.values,
                   cmap=cmap,
                   norm=norm,
                   s=10,
                   transform=ccrs.PlateCarree(),
                   )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(
                            sm,
                            ax=ax,
                            orientation="vertical",
                            fraction=0.046,
                            pad=0.04,
                            shrink=0.8,
                            )
        cbar.set_label(colorbar_label, fontsize=11)

        ax.set_title(f"{title_prefix} – {target_date}", fontsize=13)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.set_extent([-75, -30, -35, 6], crs=ccrs.PlateCarree())

        grid = ax.gridlines(
                            draw_labels=True,
                            linestyle=":",
                            color="gray",
                            alpha=0.7,
                            )
        grid.top_labels = False
        grid.right_labels = False

        filename = f"{filename_prefix}_{target_date}.png"
        filepath = AGENTS_FOLDER / filename

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def compute_state_statistics(
                                 self,
                                 shapefile: gpd.GeoDataFrame,
                                 target_date: str,
                                 filename_prefix: str="state_statistics"
                                 ):
        """
        Compute state-level heatwave statistics from predicted heatwave conditions.

        This method clips the predicted heatwave condition field and the corresponding
        monthly climatology to each polygon in the provided shapefile and computes
        summary statistics for each administrative unit.

        Statistics include mean, median, 90th percentile, maximum values, affected
        area fraction, and a categorical severity level derived from climatological
        thresholds.

        Parameters
        ----------
        shapefile : geopandas.GeoDataFrame
            GeoDataFrame containing polygon geometries and a column named 'SIGLA_UF'
            identifying each administrative unit.

        target_date : str
            Target date associated with the prediction, used to determine the
            corresponding climatological month and for output file naming.

        filename_prefix : str, optional
            Prefix used when saving the output JSON file containing the statistics.
            Default is "state_statistics".

        Raises
        ------
        ValueError
            If the target_date is not provided.

        Notes
        -----
        The resulting statistics are saved as a JSON file, containing one entry per
        administrative unit, including both predicted and climatological reference
        metrics as well as a qualitative severity classification.
        """

        if target_date is None:
            raise ValueError("target_date must be provided (YYYY-MM-DD).")

        da_pred = self.dhw_prediction.copy()

        month = pd.to_datetime(target_date).month
        clim_month = self.clim.sel(month=month)

        if not da_pred.rio.crs:
            da_pred = da_pred.rio.write_crs(4326)

        if not clim_month.rio.crs:
            clim_month = clim_month.rio.write_crs(4326)

        os.makedirs(AGENTS_FOLDER, exist_ok=True)

        results = {}

        for _, row in shapefile.iterrows():
            uf = row["SIGLA_UF"]
            geom = row.geometry

            pred_masked = da_pred.rio.clip([geom], all_touched=True, drop=True)
            clim_masked = clim_month.rio.clip([geom], all_touched=True, drop=True)

            pred_vals = pred_masked.values
            clim_vals = clim_masked.values

            pred_vals = pred_vals[np.isfinite(pred_vals)]
            clim_vals = clim_vals[np.isfinite(clim_vals)]

            if len(pred_vals) == 0 or len(clim_vals) < 5:
                continue

            mean_days = float(np.mean(pred_vals))
            median_days = float(np.median(pred_vals))
            p90_days = float(np.percentile(pred_vals, 90))
            max_days = float(np.max(pred_vals))
            fraction_area = float(np.mean(pred_vals > 0))

            clim_mean = float(np.mean(clim_vals))
            clim_median = float(np.median(clim_vals))
            clim_p90 = float(np.percentile(clim_vals, 90))
            clim_fraction_area = float(np.mean(clim_vals > 0))

            thresholds = np.percentile(clim_vals, [20, 40, 60, 80])

            if mean_days < thresholds[0]:
                severity = "very low"
            elif mean_days < thresholds[1]:
                severity = "low"
            elif mean_days < thresholds[2]:
                severity = "medium"
            elif mean_days < thresholds[3]:
                severity = "high"
            else:
                severity = "very high"

            results[uf] = {
                           "reference_period": "1961–1990",
                           "month": month,
                           "mean_days": mean_days,
                           "climatology_mean_days": clim_mean,
                           "median_days": median_days,
                           "climatology_median_days": clim_median,
                           "p90_days": p90_days,
                           "climatology_p90_days": clim_p90,
                           "max_days": max_days,
                           "fraction_area": fraction_area,
                           "climatology_fraction_area": clim_fraction_area,
                           "severity_level": severity,
                           }

        filename = f"{filename_prefix}_{target_date}.json"
        filepath = AGENTS_FOLDER / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)