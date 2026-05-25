# Function-Valued Causal Influence in Nonlinear Time Series
### ICML 2026

**Valentina Kuskova, Dmitry Zaytsev, Michael Coppedge**  
Lucy Family Institute for Data & Society / Department of Political Science, University of Notre Dame

> Paper: *Function-Valued Causal Influence in Nonlinear Time Series* (ICML 2026, Paper #28194)  
> DOI: [link to be added after proceedings publication]  
> Code: https://github.com/vkuskova/ICML2026-28194

---

## Overview

This repository contains the code and data used to produce all figures and tables in the paper.
The core contribution is a framework that estimates **function-valued causal influence**
from trained additive autoregressive models using Individual Conditional Expectation (ICE),
revealing regime-dependent causal mechanisms that scalar causal scores cannot capture.

The code is organized as a Google Colab notebook and is designed to run in that environment.
It is **not** a reproducibility package — it is provided so that others can understand,
adapt, and build on our methodology.

---

## Repository Contents

| File | Description | Origin |
|---|---|---|
| `README.md` | This file | — |
| `ICML_28194_code.ipynb` | Main Colab notebook — all figures and tables | This paper |
| `requirements.txt` | Python dependencies | Bussmann et al. (2021) |
| `NAVAR.py` | NAVAR model architecture | Bussmann et al. (2021) |
| `train_NAVAR.py` | NAVAR training loop | Bussmann et al. (2021), modified¹ |
| `dataloader.py` | Data loading utilities | Bussmann et al. (2021), modified¹ |
| `evaluate.py` | Evaluation utilities | Bussmann et al. (2021) |
| `run_NAVAR.py` | NAVAR run script | Bussmann et al. (2021), modified¹ |
| `edge_ablation.py` | Forecast-necessity edge ablation | This paper |
| `my_data.csv` | V-Dem panel data (139 countries, 35 years, 16 variables) | V-Dem v15² |

¹ `train_NAVAR.py`, `dataloader.py`, and `run_NAVAR.py` include modifications
for unbalanced panel support and NumPy/return-signature compatibility.
See the Code Attribution section below for details.
All other NAVAR files are used as released by Bussmann et al. (2021).

² See Data section below for citation and license terms.

---

## Code Attribution

The core NAVAR architecture and training code (`NAVAR.py`, `train_NAVAR.py`,
`dataloader.py`, `evaluate.py`, `run_NAVAR.py`) was originally developed by:

> Bussmann, B., Nys, J., & Latré, S. (2021). Neural additive vector autoregression
> models for causal discovery in time series. *Discovery Science*, Lecture Notes in
> Computer Science, vol 12986. Springer, Cham.
> DOI: [10.1007/978-3-030-88942-5_27](https://doi.org/10.1007/978-3-030-88942-5_27)

The original code is available at [github.com/bartbussmann/NAVAR](https://github.com/bartbussmann/NAVAR).

**Modifications made for this paper** (files dated after November 2025):

- `train_NAVAR.py` — added GPU device handling; modified return signature to include
  the trained model object (4-value return); adapted loss computation for unbalanced
  panel time series (variable-length country segments); added validation loss
  recomputation pass

- `dataloader.py` — three modifications:
  - Extended `split_timeseries` to accept a list/array of per-unit lengths for
    **unbalanced panels** (original code assumed all segments equal length); computes
    segment starts via cumsum and filters windows crossing any boundary
  - Added guard: raises `ValueError` if variable-length `split_timeseries` is passed
    with `lstm=True`, which is unsupported (would require padding/masks)
  - Fixed `np.int` → `int` in `split_train_val()` for NumPy ≥ 1.24 compatibility
    (`np.int` was removed in NumPy 1.24)

- `run_NAVAR.py` — updated return value unpack from 3 values to 4 values to match
  the modified `train_NAVAR` signature (`score_matrix, _, _, _ = train_NAVAR(...)`)

- `edge_ablation.py` — new file implementing forecast-necessity testing via
  Diebold-Mariano edge ablation (Kuskova et al., 2026, FLAIRS)

- `navar_tvnar_pipeline.py`, `tvNAR.py` — new files implementing the NAVAR → tvNAR
  dynamic causal inference pipeline (Zaytsev et al., 2026, KDD)

---

## Setup and Usage

### Requirements

- Google Colab (recommended) or a local Python 3.10+ environment
- GPU is optional but significantly speeds up training

### Google Drive folder structure

The notebook reads code and data from Google Drive. Create the following
structure before running:

```
MyDrive/
└── ICML 28194/
    ├── code/          ← upload all .py files from this repo here
    ├── data/
    │   └── my_data.csv   ← included in this repo; upload here
    └── results/       ← outputs will be saved here automatically
```

### Running the notebook

1. Upload `ICML_28194_code.ipynb` to Colab (File → Upload notebook)
2. Upload all `.py` files from this repo to `MyDrive/ICML 28194/code/`
3. Upload `my_data.csv` to `MyDrive/ICML 28194/data/`
4. Run cells in order, starting with **Cell 0** (Drive mount)
5. After **Cell 1** (dependency install), restart the Colab runtime before continuing

### ⚠️ Important: Colab-specific notes

- This notebook is designed for **Google Colab**. It uses `google.colab.drive` for
  Drive mounting and assumes Colab's default Python/CUDA environment.
- After running Cell 1, you must **restart the runtime** (Runtime → Restart session)
  before running subsequent cells. This resolves NumPy/pandas binary compatibility issues.
- All outputs (figures, CSVs) are saved automatically to `MyDrive/ICML 28194/results/`.

---

## Notebook Structure

| Cell | Content | Paper location |
|---|---|---|
| 0 | Drive mount and path setup | — |
| 1 | Dependency installation | — |
| 2 | Reproducibility seed | — |
| 3 | Imports | — |
| 4 | Causal mechanisms and synthetic data generator | Section 5.1, Appendix B |
| 5 | NAVAR config and training wrapper | Appendix A |
| 6 | Synthetic experiments (scalar scores) | Table 1 |
| 7 | Model forward helpers (ICE utilities) | Section 4 |
| 8 | ICE response functions — synthetic | Figure 2 |
| 9 | Load democracy data, train NAVAR, causal matrix heatmap | Section 5.2, Appendix E |
| 10 | Democracy panel ICE helpers | Section 4 |
| 11 | ICE response functions — democracy | Figure 3 |
| 12a | Bootstrap CI computation (run once, cached) | Appendix G |
| 12b | Bootstrap CI figure (re-run freely) | Appendix G |
| 13 | ICE recovery metrics — synthetic | Appendix H |
| 14 | Lag-specific ICE analysis | Appendix I |
| 15 | Prevalence of nonlinear functional form | Appendix J |

---

## Data

### Synthetic data
Generated programmatically in Cell 4. No external data needed.

### Democracy panel data (`my_data.csv`)

The empirical analysis uses 16 components from the
**Varieties of Democracy (V-Dem) v15** dataset:

> Coppedge, Michael, et al. (2025). *V-Dem Country-Year Dataset v15*.
> Varieties of Democracy (V-Dem) Project.
> DOI: [10.23696/vdemds25](https://doi.org/10.23696/vdemds25)

The included `my_data.csv` is a preprocessed subset: 139 countries with complete
coverage across all 16 selected variables for 1990–2024 (35 years), standardized
to zero mean and unit variance. The 16 variables are listed in Appendix E,
Table E.1 of the paper. V-Dem data is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by the V-Dem Institute;
our preprocessed subset is distributed under the same terms.

---

## Key Outputs

All outputs are saved to `MyDrive/ICML 28194/results/`:

| File | Description |
|---|---|
| `synthetic_navar_scores.csv` | Scalar causal scores across 60 runs (Table 1) |
| `navar_causal_matrix.csv` | Democracy NAVAR causal score matrix |
| `Figure2_synthetic_ICE.png` | ICE response functions — four mechanisms |
| `Figure3_democracy_ICE.png` | ICE response functions — democracy edges |
| `navar_causal_matrix_heatmap.png` | Heatmap of democracy causal matrix |
| `bootstrap_ice_results.npy` | Cached bootstrap results (Appendix G) |
| `Figure_ICE_bootstrap_CI.png` | Bootstrap confidence bands (Appendix G) |
| `synthetic_recovery_metrics.csv` | Pearson r recovery table (Appendix H) |
| `Figure_recovery_metrics.png` | Recovery quality bar chart (Appendix H) |
| `Figure_AppendixI_lag_specific_ICE.png` | Lag-specific ICE curves (Appendix I) |
| `democracy_heterogeneity_summary.csv` | Edge classification results (Appendix J) |
| `Figure_democracy_heterogeneity.png` | Heterogeneity prevalence chart (Appendix J) |

---

## Related Work

This code is part of a broader research program on nonlinear causal inference
in panel time series:

- **FLAIRS 2026** — Forecast-necessity testing via Diebold-Mariano edge ablation:
  Kuskova, V., Zaytsev, D., & Coppedge, M. (2026). *Beyond Coefficients: Forecast-Necessity
  Testing for Interpretable Causal Discovery in Nonlinear Time-Series Models.*
  DOI: [10.32473/flairs.39.1.141791](https://doi.org/10.32473/flairs.39.1.141791)

- **KDD 2026** — Dynamic causal inference using NAVAR-derived structural priors:
  Zaytsev, D., Kuskova, V., & Coppedge, M. (2026). *From Causal Discovery to Dynamic
  Causal Inference in Neural Time Series.* arXiv:2603.20980

- **Trajectory-aware reliability modeling:**
  Zaytsev, D., Kuskova, V., & Coppedge, M. (2026). *Trajectory-Aware Reliability
  Modeling of Democratic Systems.* arXiv:2604.20127

---

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{Kuskova2026functionvalued,
  title     = {Function-Valued Causal Influence in Nonlinear Time Series},
  author    = {Kuskova, Valentina and Zaytsev, Dmitry and Coppedge, Michael},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

If you use or adapt the NAVAR code, please also cite:

```bibtex
@inproceedings{Bussmann2021navar,
  title     = {Neural Additive Vector Autoregression Models for Causal Discovery in Time Series},
  author    = {Bussmann, Bart and Nys, Jannes and Latr{\'e}, Steven},
  booktitle = {Discovery Science},
  year      = {2021},
  publisher = {Springer},
  doi       = {10.1007/978-3-030-88942-5_27}
}
```

---

## License

- **Our code** (`edge_ablation.py`, `navar_tvnar_pipeline.py`, `tvNAR.py`,
  `ICML_28194_code.ipynb`, and all modifications to NAVAR files): **MIT License**
- **Original NAVAR code** (`NAVAR.py`, `dataloader.py`, `evaluate.py`, `run_NAVAR.py`,
  base `train_NAVAR.py`): retains its original license from Bussmann et al. (2021)
- **V-Dem data** (`my_data.csv`): **CC BY 4.0** — V-Dem Institute

---

## Contact

Valentina V. Kuskova — vkuskova@nd.edu  
Lucy Family Institute for Data & Society, University of Notre Dame
