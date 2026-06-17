# Baseline Prediction Framework for Wind and PV Power Forecasting

This repository contains the benchmark code for the paper:

**Review and Assessment of Data-Driven Wind and Solar Forecasting Models: Methodology, Application, Benchmarking, and Analysis**

The code has been cleaned to focus on reproducing the benchmark experiments reported in the paper, especially Tables 6-9. It provides a unified evaluation pipeline for statistical models, classical machine learning models, deep time-series models, LLM-based models, a multimodal Time-VLM model, and Chronos.

## What This Repository Reproduces

The current codebase is scoped to long-term forecasting experiments on wind power and PV power datasets.

| Paper result | Script | Description |
| --- | --- | --- |
| Table 6 | `benchmark_all_models.py` | Predictive accuracy on the Wind Power dataset for `T = 1, 16, 96` |
| Table 7 | `benchmark_all_models.py` | Predictive accuracy on the PV Power dataset for `T = 1, 16, 96` |
| Table 8 | `benchmark_all_models.py` | Efficiency and deployment cost on the Wind Power dataset at `T = 96` |
| Table 9 | `benchmark_noise_t1.py` | Robustness under 10% Gaussian white noise on test inputs at `T = 1` |

The benchmark includes the following models:

- Statistical models: `ARIMA`, `ETS`
- Machine learning models: `RandomForest`, `SVR`
- Recurrent neural networks: `LSTM`, `GRU`
- Advanced deep forecasting models: `DLinear`, `TimesNet`, `Informer`, `PatchTST`, `iTransformer`
- LLM, multimodal, and foundation models: `Time-LLM`, `Time-VLM`, `Chronos`

## Repository Structure

```text
.
|-- benchmark_all_models.py        # Main clean benchmark for Tables 6-8
|-- benchmark_noise_t1.py          # Noise robustness benchmark for Table 9
|-- run.py                         # Single neural-model runner used by benchmark scripts
|-- requirements.txt               # Python environment snapshot
|-- data_provider/                 # Dataset loader and data factory
|-- exp/                           # Long-term forecasting experiment logic
|-- layers/                        # Shared neural network layers
|-- models/                        # Forecasting model implementations
|-- src/TimeVLM/                   # Time-VLM implementation and CLIP manager
|-- utils/                         # Metrics, tools, time features, DM test
`-- dataset/prompt_bank/custom.txt # Prompt template used by Time-LLM/Time-VLM
```

Only the long-term forecasting task is kept in this cleaned version. Other unrelated tasks and models have been removed to keep the release aligned with the four benchmark tables.

## Environment Setup

The experiments were developed for Linux servers with NVIDIA GPUs. A typical setup is:

```bash
conda create -n energy-benchmark python=3.10 -y
conda activate energy-benchmark

# Install PyTorch according to your CUDA version first if needed.
# Then install the project dependencies.
pip install -r requirements.txt

# Extra packages required by the statistical and Chronos baselines.
pip install statsmodels chronos-forecasting
```

Notes:

- `requirements.txt` is an environment snapshot from the original server. If a CUDA-specific PyTorch wheel does not match your server, install the correct `torch`, `torchvision`, and `torchaudio` versions from the official PyTorch index, then install the remaining packages.
- `ARIMA` and `ETS` require `statsmodels`.
- `Chronos` requires `chronos-forecasting` and loads `amazon/chronos-t5-small`.
- `Time-LLM` uses Hugging Face models. The default setting is GPT-2: `openai-community/gpt2`.
- `Time-VLM` currently keeps only the CLIP branch and loads CLIP from a local path in `src/TimeVLM/vlm_manager.py`:

```python
clip_path = '/root/phr/ICML25-TimeVLM-main/clip-vit-base-patch32'
```

If your CLIP checkpoint is stored elsewhere, edit this path before running `Time-VLM`.

## Data Preparation

The paper uses two stations from the Chinese State Grid Renewable Energy Generation Forecasting Competition dataset:

- Wind power dataset: 2019-2020, 15-minute resolution, 70,176 data points.
- PV power dataset: January 1, 2019 to February 22, 2019, 15-minute resolution, 5,000 data points.

Place the downloaded files under `dataset/`, for example:

```text
dataset/
|-- wind.csv
`-- PV.csv
```

The benchmark scripts accept `.csv`, `.xls`, or `.xlsx` files. Each input file must contain:

- One time column whose name contains `time` or `date`.
- One target column whose name contains `power`.
- Optional exogenous feature columns.

The scripts automatically standardize each file into:

```text
date, <exogenous features...>, power
```

Standardized copies are written to:

```text
<out_root>/prepared_data/
```

The data split is fixed to `8:1:1` for train, validation, and test.

## Reproduce Tables 6-8

Run the clean benchmark:

```bash
python benchmark_all_models.py \
  --repo_root . \
  --run_py run.py \
  --wind_path ./dataset/wind.csv \
  --pv_path ./dataset/PV.csv \
  --out_root ./benchmark_master \
  --seq_len 288 \
  --label_len 144 \
  --pred_lens 1 16 96 \
  --gpu 0
```

Main outputs:

```text
benchmark_master/
|-- prepared_data/
|-- benchmark_results/
|-- summary_all_results.csv
|-- summary_accuracy.csv
|-- summary_efficiency.csv
`-- summary_dm_test.csv
```

Use `summary_accuracy.csv` to reconstruct Tables 6 and 7. The key columns are:

```text
model, dataset, pred_len, mae, mse, rmse, r2
```

Use `summary_efficiency.csv` to reconstruct Table 8. Filter it to:

```text
dataset == wind
pred_len == 96
```

The efficiency columns are:

```text
params_k, mem_mb, t_train_s_per_ep, t_inf_ms
```

## Reproduce Table 9

Table 9 requires the clean `T = 1` results from `summary_accuracy.csv`. Run `benchmark_all_models.py` first, then run:

```bash
python benchmark_noise_t1.py \
  --repo_root . \
  --run_py run.py \
  --wind_path ./dataset/wind.csv \
  --pv_path ./dataset/PV.csv \
  --clean_summary_path ./benchmark_master/summary_accuracy.csv \
  --out_root ./benchmark_noise_t1_gwn10 \
  --seq_len 288 \
  --label_len 144 \
  --pred_len 1 \
  --noise_level 0.10 \
  --gpu 0 \
  --seed 2024
```

Main outputs:

```text
benchmark_noise_t1_gwn10/
|-- prepared_data/
|-- benchmark_results/
|-- summary_noisy_results.csv
`-- robustness_drop_t1.csv
```

Use `robustness_drop_t1.csv` to reconstruct Table 9. The drop percentage is computed as:

```text
(Noisy - Clean) / Clean * 100%
```

Negative values mean the noisy-test result is slightly better than the clean-test result.

Implementation note: the classical and Chronos branches perturb test input windows before model prediction and clip raw noisy values to be non-negative. The neural-model branch injects reproducible Gaussian noise inside the test dataloader path using normalized test tensors.

## Run One Neural Model Manually

The benchmark scripts are the recommended way to reproduce the paper tables. For debugging, you can also run a single neural model with `run.py`.

Example after `benchmark_all_models.py` has generated standardized data:

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model DLinear \
  --data custom \
  --root_path ./benchmark_master/prepared_data \
  --data_path wind_std.csv \
  --features MS \
  --target power \
  --freq t \
  --seq_len 288 \
  --label_len 144 \
  --pred_len 96 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 1 \
  --d_model 128 \
  --d_ff 768 \
  --train_epochs 10 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --inverse \
  --gpu 0 \
  --result_root ./debug_results
```

Adjust `enc_in` and `dec_in` to the number of non-date columns in the standardized CSV.

Supported neural model names in `run.py`:

```text
LSTM, GRU, DLinear, TimesNet, Informer, PatchTST, iTransformer, TimeLLM, TimeVLM
```

## Result Files

Each model run writes a result directory under the configured `result_root`. Typical files include:

```text
metrics.json      # Main metrics and efficiency values
metrics.npy       # [MAE, SSE, MSE, RMSE, R2]
pred.npy          # Predictions
true.npy          # Ground truth
run_report.txt    # Human-readable run summary
run_stdout.log    # Captured stdout/stderr from benchmark orchestration
```

The benchmark scripts collect these per-run files into summary CSV files for table generation.

## Metrics

Predictive accuracy:

- `MAE`
- `MSE`
- `RMSE`
- `R2`

Efficiency and deployment cost:

- `Params (K)`: number of trainable parameters in thousands.
- `Mem (MB)`: maximum GPU memory usage during neural-model training/inference.
- `t_train (s/ep)`: average training time per epoch.
- `t_inf (ms)`: average inference latency per sample.

Traditional non-neural baselines may not report parameter or GPU-memory metrics.

## Reproducibility Notes

- The default forecasting setup uses `seq_len = 288`, `label_len = 144`, and prediction horizons `T = 1, 16, 96`.
- The target column is always normalized to `power` after preprocessing.
- The train, validation, and test split is chronological.
- Deep model results can vary slightly across GPUs, CUDA versions, and PyTorch versions.
- Chronos is used as a zero-shot univariate forecaster. It uses only historical `power` values as context and ignores exogenous features.
- The current cleaned runner supports only `long_term_forecast`.

## Citation

If you use this repository, please cite the accompanying paper:

```bibtex
@article{fan2026review,
  title   = {Review and Assessment of Data-Driven Wind and Solar Forecasting Models: Methodology, Application, Benchmarking, and Analysis},
  author  = {Fan, Hang and Liu, Weican and Pei, Haoran and Zhang, Zuhan and Liu, Dunnan and Cheng, Long and Xu, Xiaomin},
  year    = {2026}
}
```

## License

No license file is included in this cleaned repository yet. Add a license before public release if the code will be distributed publicly.
