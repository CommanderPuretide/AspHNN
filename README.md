# ASpHNN — Attention-Augmented Stochastic Port-Hamiltonian Neural Networks

Reference implementation of **ASpHNN**, a stability-aware neural architecture for
time-series forecasting, as described in the paper
*“A Stability-Aware Neural Network Architecture for Time Series Forecasting.”*

ASpHNN couples a **stochastic port-Hamiltonian backbone** (for physically
structured, energy-aware dynamics) with an **attention-based residual path** (for
long-range temporal dependencies) and a **learnable diffusion head** (for
state-dependent uncertainty). The state is advanced with an Euler–Maruyama /
midpoint rollout, and the model is trained with a combination of one-step,
moment-matching (drift + diffusion), energy-consistency, and structure /
diffusion regularization losses.

In this codebase the model is implemented under the name **`PHEnergyPatch`**
(Port-Hamiltonian + Energy + Patch residual).

## Mapping: paper → code

| Paper component | Where it lives |
| --- | --- |
| Port-Hamiltonian drift $f_{\text{phys}}=(J-R)\nabla H_\phi$, with $J=A-A^\top$, $R=BB^\top$ | `HamiltonianMLP` + drift assembly in `models/PHEnergyPatch.py` |
| Attention residual $f_{\text{attn}}$ (PatchTST-style + linear shortcut) | `ResidualPatchTST` in `models/PHEnergyPatch.py`, built on `layers/PatchTST_backbone.py` |
| Learnable diffusion head $g_\psi$, $\Sigma=g_\psi g_\psi^\top$ | diffusion module in `models/PHEnergyPatch.py` (`--pha_diff_rank`, `--pha_diff_hidden`, `--pha_diff_scale`) |
| Euler–Maruyama / implicit-midpoint rollout | `--pha_dt`, `--pha_midpoint_iters`, `--pha_midpoint_tol` |
| Loss weights $\lambda_e,\lambda_s,\lambda_m,\lambda_r,\lambda_{\text{attn}}$ | `--pha_lambda_energy/struct/moment/diff_reg/attn` |
| Adaptive loss weighting (uncertainty / EMA-norm) | `--pha_loss_weighting`, `--pha_loss_ema_*`, `--pha_loss_weight_min/max` |
| Component ablations (w/o Hamiltonian / Attention / Diffusion) | `--pha_use_hamiltonian`, `--pha_use_attention`, `--pha_use_diffusion` |
| Multivariate decomposition variant | `models/PHEnergyPatchMulti.py` |

## Repository structure

```
run_longExp.py            # entry point: argument parsing + experiment launch
exp/
  exp_basic.py            # base experiment class (device setup)
  exp_main.py             # training / validation / test + autoregressive rollout
models/
  PHEnergyPatch.py        # ASpHNN model (main)
  PHEnergyPatchMulti.py   # ASpHNN multivariate (series-decomposition) variant
  DLinear.py              # dependency only: provides series_decomp used by *Multi
layers/
  PatchTST_backbone.py    # attention residual backbone
  PatchTST_layers.py      # backbone building blocks
  RevIN.py                # reversible instance normalization
data_provider/
  data_factory.py         # dataset/loader factory
  data_loader.py          # ETT / custom CSV dataset classes
utils/
  timefeatures.py         # time-covariate features (used by the model)
  tools.py                # EarlyStopping, LR scheduling, plotting, FLOPs
  metrics.py              # MSE/MAE/RMSE/MAPE/MSPE
scripts/
  PHEnergyPatch/          # per-dataset run scripts (main model)
  PHEnergyPatchMulti/     # per-dataset run scripts (multivariate variant)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

Place the standard long-term-forecasting CSVs under `./dataset/`, e.g.
`./dataset/weather.csv`, `ETTm1.csv`, `electricity.csv`, `traffic.csv`,
`exchange_rate.csv`. These are the public benchmarks used by Autoformer/Informer
and are not redistributed here.

## Running

Use the provided scripts (run from the repository root), or call `run_longExp.py`
directly. Example (Weather):

```bash
bash scripts/PHEnergyPatch/weather.sh
```

```bash
python -u run_longExp.py \
    --is_training 1 --model PHEnergyPatch \
    --root_path ./dataset/ --data_path weather.csv --data custom \
    --features M --seq_len 32 --pred_len 8 --enc_in 21 \
    --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
    --pha_use_hamiltonian 1 --pha_use_attention 1 --pha_use_diffusion 1 \
    --pha_lambda_energy 0.1 --pha_lambda_struct 0.1 --pha_lambda_moment 0.1 \
    --pha_lambda_diff_reg 0.1 --pha_dt 0.01 --pha_loss_weighting uncertainty \
    --train_epochs 10 --batch_size 32 --learning_rate 1e-4 --itr 1
```

See `run_longExp.py` for the full list of `--pha_*` model hyperparameters.

## Notes

- The synthetic Lorenz-96 ablation and Optuna hyperparameter search reported in
  the paper were produced with separate driver code and are **not** included in
  this repository, which targets the model and the real-benchmark pipeline.
- This project reuses data-loading, normalization (RevIN), and the PatchTST
  attention backbone from the PatchTST codebase
  (https://github.com/yuqinie98/PatchTST); see `LICENSE`. Baseline models
  (Autoformer, Informer, Transformer, (N/D)Linear, vanilla PatchTST) are **not**
  bundled — only files required by ASpHNN are included.
  
  
  "The strength of your force may be calculated by multiplying its weight by its velocity..." – The Forty Second Meditation on the Way of the Warrior, Commander Puretide, Fire Caste, Tau Empire.
```
