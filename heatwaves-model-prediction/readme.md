# heatwaves-model-prediction

This folder documents the research and training workflow used to develop the **Temporal U-Net + Transformer** model for forecasting monthly heatwave-condition anomalies over Brazil.

The notebook in this directory represents the experimental phase of the project and contains:

- Data preprocessing
- Sequence construction
- Model architecture definition
- Training loop
- Validation and testing
- Baseline comparison
- Skill score computation
- Qualitative spatial error analysis

This folder is not required for production inference, but it defines how the deployed model checkpoint was generated.

---

## 1. Notebook Overview

Main file:

```
WMO_HWC_Model.ipynb
```

The notebook was developed and executed in Google Colab and uses the GPU acceleration of the free tier licence.

---

## 2. Dataset Preparation

### 2.1 Input Dataset

Source:

```
WMO_heatwave_conditions_1961-present.nc
```

Variable used:
- `HWC` (monthly heatwave-condition day counts)

Processing steps:

1. Remove final incomplete timestep.
2. Compute monthly climatology (1961–1990).
3. Derive monthly anomalies and apply z-score standardization.
4. Convert to NumPy arrays.
5. Construct sliding sequences of length 12 months.

Resulting supervised format:

- X: (N, 12, 1, H, W)
- y: (N, 1, H, W)

---

## 3. Normalization Strategy

A global Z-score normalization was applied:

- μ = global mean of training set
- σ = global standard deviation of training set

Normalization applied to:
- Inputs (X)
- Targets (y)

The same μ and σ values are stored in the checkpoint and reused during inference.

---

## 4. Model Architecture

### 4.1 Temporal U-Net + Transformer

The architecture combines:

- U-Net (ResNet18 encoder backbone)
- ResNet18 was chosen for simplicity
- Transformer encoder at the bottleneck

Design principles:

- U-Net captures spatial patterns.
- Transformer models temporal dependencies across 12-month sequences.
- Each spatial location is treated as a temporal sequence during attention computation.

Input shape:
- (Batch, Time, Channel, Latitude, Longitude)

Output:
- Predicted anomaly field for next month.

---

## 5. Training Configuration

Hyperparameters:

- Learning rate: 1e-4
- Optimizer: AdamW
- Weight decay: 1e-5
- Loss function: MSE
- Scheduler: ReduceLROnPlateau
- Early stopping patience: 7 epochs
- Batch size: 1
- Epochs: 50 (with early stopping)

Training/validation/test split:

- 80% training
- 10% validation
- 10% testing
- No shuffling (temporal integrity preserved)

---

## 6. Training Results

Early stopping occurred at epoch 13.

Validation metrics:

- Train MSE: 0.614720
- Val MSE: 1.819998
- Val RMSE: 1.349073

Test metrics:

- Test RMSE (normalized): 1.572304
- Test RMSE (physical units, days): 2.629446

---

## 7. Baseline Comparison

Baseline model:
- Monthly climatology (1961–1990)

Results:

- MSE baseline: 8.334930
- MSE model: 6.913987
- Skill score: 0.170

Skill score definition:

Skill = 1 − (MSE_model / MSE_baseline)

Interpretation:
- The model outperforms climatology.
- Improvement is moderate but statistically meaningful.

---

## 8. Spatial Error Characteristics

From residual maps:

```
(da_true - da_pred)
```

Observations:

- Larger errors in southern Brazil.
- Increased errors in regions with higher topography.
- Spatial heterogeneity not fully captured by monthly-scale training.

These patterns suggest:

- Topographic influence is not explicitly modeled.
- Monthly aggregation smooths important short-term variability.
- Model complexity may not fully compensate for reduced temporal resolution.

---

## 9. Limitations of Monthly Training

The model was trained using **monthly data** due to computational limitations of the free Google Colab environment.

Consequences:

- Limited number of training samples.
- Reduced temporal variability.
- Reduced representation of short-duration extremes.
- Lower effective model generalization capacity.

---

## 10. Recommended Improvement Strategy

For a production-grade workflow, the following strategy is recommended:

1. Train the same architecture using **daily-resolution data**.
   - Increased sample size.
   - Improved representation of variability.
   - Better capture of extreme-event dynamics.

2. Save checkpoint from daily training.

3. Fine-tune the model on monthly anomalies:
   - Initialize weights from daily-trained model.
   - Reduce learning rate.
   - Perform domain adaptation for monthly scale.

This approach would:

- Improve feature extraction robustness.
- Improve convergence stability.
- Potentially increase skill score.
- Preserve architectural consistency with deployed inference pipeline.

---

## 11. Reproducibility Notes

- Device auto-detection: CUDA → MPS → CPU.
- Global normalization parameters stored in checkpoint.
- Sequence length fixed at 12 months.
- Chronological split ensures no data leakage.

The exported checkpoint:

```
best_temporal_unet_transformer.pt
```

Contains:

- model_state
- mu
- sigma

This file is copied to:

```
inference_api/model/
```

for production inference.

---

## 12. Summary

This folder documents the research phase that led to the deployed model.

Key characteristics:

- Spatiotemporal deep learning architecture.
- Monthly anomaly prediction.
- Deterministic reproducibility.
- Skill superior to climatological baseline.
- Clear spatial error structure.
- Identified pathway for performance improvement via daily-scale pretraining.

This training notebook forms the scientific foundation of the inference layer used in AgenticAI Heatwaves BR.