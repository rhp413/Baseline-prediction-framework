import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from utils.metrics import metric_extended
try:
    from chronos import ChronosPipeline
    HAS_CHRONOS = True
except Exception:
    HAS_CHRONOS = False
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]+', '_', str(s)).strip('_')


def read_table(path: str) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith('.xlsx') or path.lower().endswith('.xls'):
        return pd.read_excel(path)
    return pd.read_csv(path)


def detect_time_col(columns):
    for c in columns:
        lc = str(c).lower()
        if 'time' in lc or 'date' in lc:
            return c
    raise ValueError(f'未检测到时间列，当前列名：{list(columns)}')


def detect_target_col(columns):
    for c in columns:
        lc = str(c).lower()
        if 'power' in lc:
            return c
    raise ValueError(f'未检测到目标列(power)，当前列名：{list(columns)}')


def standardize_energy_file(in_path: str, out_dir: str) -> dict:
    df = read_table(in_path).copy()
    time_col = detect_time_col(df.columns)
    target_col = detect_target_col(df.columns)

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    rename_map = {time_col: 'date', target_col: 'power'}
    for c in df.columns:
        if c in rename_map:
            continue
        rename_map[c] = safe_name(str(c).lower())

    df = df.rename(columns=rename_map)
    feature_cols = [c for c in df.columns if c not in ['date']]
    df = df[['date'] + [c for c in feature_cols if c != 'power'] + ['power']]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / (Path(in_path).stem + '_std.csv')
    df.to_csv(out_csv, index=False)

    return {
        'input_path': str(in_path),
        'csv_path': str(out_csv),
        'dataset_name': Path(in_path).stem,
        'enc_in': len(df.columns) - 1,
        'dec_in': len(df.columns) - 1,
        'c_out': 1,
    }

def build_windows_from_standard_csv(csv_path: str, seq_len: int, pred_len: int):
    df = pd.read_csv(csv_path)
    assert 'date' in df.columns and 'power' in df.columns

    values = df.drop(columns=['date']).values.astype(np.float64)
    feature_names = list(df.drop(columns=['date']).columns)
    target_idx = feature_names.index('power')

    n = len(df)
    num_train = int(n * 0.8)
    num_test = int(n * 0.1)
    num_val = n - num_train - num_test

    border1s = [0, num_train - seq_len, num_train + num_val - seq_len]
    border2s = [num_train, num_train + num_val, n]

    splits = {}
    for split_name, split_id in [('train', 0), ('val', 1), ('test', 2)]:
        b1 = border1s[split_id]
        b2 = border2s[split_id]
        split_data = values[b1:b2]

        X, Y = [], []
        for i in range(len(split_data) - seq_len - pred_len + 1):
            x = split_data[i:i + seq_len]
            y = split_data[i + seq_len:i + seq_len + pred_len, target_idx]
            X.append(x)
            Y.append(y)

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        splits[split_name] = (X, Y)

    meta = {
        'feature_names': feature_names,
        'target_idx': target_idx,
        'num_train': num_train,
        'num_val': num_val,
        'num_test': num_test,
    }
    return splits, meta


def add_gaussian_noise_to_input_windows(
    X_test_clean: np.ndarray,
    noise_level: float = 0.10,
    seed: int = 2024,
    clip_nonnegative: bool = True,
    include_target_history: bool = True,
    target_idx: int = None,
):
    """
    正确的鲁棒性加噪方式：
    - 只扰动模型输入 X_test；
    - 输入中的历史 power 通道也加噪；
    - future label y_test_clean 完全不动；
    - 按每个输入通道的标准差生成 10% 高斯白噪声。
    """
    rng = np.random.default_rng(seed)
    X_noisy = X_test_clean.copy()

    if X_noisy.ndim != 3:
        raise ValueError(f'X_test_clean 应为 [N, seq_len, C]，实际 shape={X_noisy.shape}')

    channel_std = X_test_clean.reshape(-1, X_test_clean.shape[-1]).std(axis=0, ddof=0)
    channel_std = np.where(channel_std < 1e-12, 0.0, channel_std)

    noise = rng.normal(
        loc=0.0,
        scale=noise_level * channel_std.reshape(1, 1, -1),
        size=X_noisy.shape
    )

    if not include_target_history:
        if target_idx is None:
            raise ValueError('include_target_history=False 时必须提供 target_idx')
        noise[:, :, target_idx] = 0.0

    X_noisy = X_noisy + noise

    if clip_nonnegative:
        X_noisy = np.clip(X_noisy, a_min=0.0, a_max=None)

    return X_noisy, channel_std




def flatten_windows(X):
    return X.reshape(X.shape[0], -1)

_CHRONOS_PIPELINE_CACHE = {}


def get_chronos_pipeline(model_id: str = "amazon/chronos-t5-small"):
    """
    Chronos-T5 zero-shot forecasting pipeline.
    使用缓存，避免同一个脚本中每个数据集/步长都重新加载模型。
    """
    if not HAS_CHRONOS:
        raise RuntimeError(
            "chronos-forecasting 未安装。请先执行：pip install chronos-forecasting"
        )

    import torch

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32

    key = (model_id, device_map, str(torch_dtype))
    if key not in _CHRONOS_PIPELINE_CACHE:
        print(f"[Chronos] loading {model_id} on {device_map}, dtype={torch_dtype}")
        pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        _CHRONOS_PIPELINE_CACHE[key] = pipeline

    return _CHRONOS_PIPELINE_CACHE[key]


def count_chronos_params_k(pipeline):
    """
    统计 Chronos 的可训练参数量，单位 K。
    """
    model = getattr(pipeline, "model", None)
    if model is None:
        return None

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1000.0


def run_chronos_forecast(test_X, pred_len: int, batch_size: int = 64, num_samples: int = 20):
    """
    Chronos-T5 是单变量 zero-shot 模型。
    这里使用每个窗口中的历史 power 序列作为 context，不使用外生变量。
    """
    import torch

    torch.manual_seed(2024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2024)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    pipeline = get_chronos_pipeline("amazon/chronos-t5-small")
    params_k = count_chronos_params_k(pipeline)

    # 你的标准化数据中 power 被放在最后一列，因此 -1 是历史目标序列
    contexts = test_X[:, :, -1].astype(np.float32)

    preds = []
    infer_start = time.perf_counter()

    for start in range(0, len(contexts), batch_size):
        end = min(start + batch_size, len(contexts))
        context_batch = torch.tensor(contexts[start:end], dtype=torch.float32)

        with torch.no_grad():
            forecast = pipeline.predict(
                context_batch,
                prediction_length=pred_len,
                num_samples=num_samples,
            )

        # forecast: [batch, num_samples, pred_len]
        median_pred = torch.quantile(forecast, q=0.5, dim=1)
        preds.append(median_pred.detach().cpu().numpy())

    infer_end = time.perf_counter()

    preds = np.concatenate(preds, axis=0)

    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
    else:
        mem_mb = None

    t_inf_ms = (infer_end - infer_start) / len(test_X) * 1000.0

    return preds, params_k, mem_mb, t_inf_ms
def save_txt_report(path: Path, info: dict):
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in info.items():
            f.write(f'{k}: {v}\n')


def tee_subprocess_to_log(cmd, cwd, env, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'w', encoding='utf-8') as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in proc.stdout:
            print(line, end='')
            lf.write(line)
            lf.flush()

        return_code = proc.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def count_model_params(model) -> float:
    if model is None:
        return None

    total = 0

    if hasattr(model, 'estimators_'):
        for est in model.estimators_:
            if hasattr(est, 'get_booster'):
                booster = est.get_booster()
                total += len(booster.get_dump())
            elif hasattr(est, 'estimators_'):
                total += len(est.estimators_)
            else:
                total += 0

        return float(total) if total > 0 else None

    return None

def evaluate_and_save(result_dir: Path, model_name: str, dataset_name: str, pred_len: int,
                      preds: np.ndarray, trues: np.ndarray,
                      params_k=None, mem_mb=None, t_train_s_per_ep=None, t_inf_ms=None):
    mae, sse, mse, rmse, mape, mspe, r2 = metric_extended(preds, trues)
    metrics = {
        'model': model_name,
        'dataset': dataset_name,
        'pred_len': int(pred_len),
        'mae': float(mae),
        'sse': float(sse),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'mspe': float(mspe),
        'params_k': None if params_k is None else float(params_k),
        'mem_mb': None if mem_mb is None else float(mem_mb),
        't_train_s_per_ep': None if t_train_s_per_ep is None else float(t_train_s_per_ep),
        't_inf_ms': None if t_inf_ms is None else float(t_inf_ms),
    }

    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / 'pred.npy', preds)
    np.save(result_dir / 'true.npy', trues)

    with open(result_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    save_txt_report(result_dir / 'run_report.txt', metrics)
    return metrics


def run_classical_baseline(model_name: str, csv_path: str, seq_len: int, pred_len: int,
                           result_root: Path, noise_level: float = 0.10,
                           seed: int = 2024, include_target_history: bool = True):
    splits, meta = build_windows_from_standard_csv(csv_path, seq_len=seq_len, pred_len=pred_len)
    X_train, y_train = splits['train']
    X_test_clean, y_test_clean = splits['test']

    X_test_noisy, channel_std = add_gaussian_noise_to_input_windows(
        X_test_clean=X_test_clean,
        noise_level=noise_level,
        seed=seed,
        clip_nonnegative=True,
        include_target_history=include_target_history,
        target_idx=meta['target_idx'],
    )

    train_series = X_train[:, -1, meta['target_idx']]
    dataset_name = Path(csv_path).stem.replace('_std', '')
    result_dir = result_root / f'{model_name}__{dataset_name}_inputgwn10__pl{pred_len}'
    result_dir.mkdir(parents=True, exist_ok=True)

    log_path = result_dir / 'train.log'
    params_k = None
    mem_mb = None

    with open(log_path, 'w', encoding='utf-8') as logf:
        logf.write(f'model: {model_name}\n')
        logf.write(f'dataset: {dataset_name}\n')
        logf.write(f'csv_path: {csv_path}\n')
        logf.write(f'seq_len: {seq_len}\n')
        logf.write(f'pred_len: {pred_len}\n')
        logf.write(f'noise_level: {noise_level}\n')
        logf.write(f'noise_seed: {seed}\n')
        logf.write(f'include_target_history: {include_target_history}\n')
        logf.write(f'label_is_clean: True\n')
        logf.write(f'feature_names: {meta["feature_names"]}\n')
        logf.write(f'target_idx: {meta["target_idx"]}\n')
        logf.write(f'channel_std: {channel_std.tolist()}\n')
        logf.write(f'X_train.shape: {X_train.shape}\n')
        logf.write(f'y_train.shape: {y_train.shape}\n')
        logf.write(f'X_test_clean.shape: {X_test_clean.shape}\n')
        logf.write(f'X_test_noisy.shape: {X_test_noisy.shape}\n')
        logf.write(f'y_test_clean.shape: {y_test_clean.shape}\n\n')
        logf.flush()

        try:
            if model_name == 'ARIMA':
                if not HAS_STATSMODELS:
                    raise RuntimeError('statsmodels 未安装，无法运行 ARIMA。')

                logf.write('config: ARIMA(order=(5,1,0)); noisy test input; clean future label\n')
                train_start = time.perf_counter()
                preds = []
                history = list(train_series.astype(float))
                infer_total = 0.0

                for i in range(len(X_test_noisy)):
                    model = ARIMA(history, order=(5, 1, 0))
                    model_fit = model.fit()

                    pred_start = time.perf_counter()
                    forecast = model_fit.forecast(steps=pred_len)
                    pred_end = time.perf_counter()

                    preds.append(np.asarray(forecast, dtype=np.float64))
                    history.append(float(X_test_noisy[i, -1, meta['target_idx']]))
                    infer_total += (pred_end - pred_start)

                train_end = time.perf_counter()
                preds = np.asarray(preds, dtype=np.float64)
                t_train_s_per_ep = train_end - train_start
                t_inf_ms = infer_total / len(X_test_noisy) * 1000.0

            elif model_name == 'ETS':
                if not HAS_STATSMODELS:
                    raise RuntimeError('statsmodels 未安装，无法运行 ETS。')

                logf.write("config: ExponentialSmoothing; noisy test input; clean future label\n")
                train_start = time.perf_counter()
                preds = []
                history = list(train_series.astype(float))
                infer_total = 0.0

                for i in range(len(X_test_noisy)):
                    try:
                        model = ExponentialSmoothing(
                            history,
                            trend='add',
                            seasonal='add',
                            seasonal_periods=96,
                            initialization_method='estimated'
                        )
                        model_fit = model.fit(optimized=True, use_brute=False)
                    except Exception as e:
                        logf.write(f'ETS seasonal fit failed at step {i}: {e}\n')
                        model = ExponentialSmoothing(
                            history,
                            trend='add',
                            seasonal=None,
                            initialization_method='estimated'
                        )
                        model_fit = model.fit(optimized=True, use_brute=False)

                    pred_start = time.perf_counter()
                    forecast = model_fit.forecast(pred_len)
                    pred_end = time.perf_counter()

                    preds.append(np.asarray(forecast, dtype=np.float64))
                    history.append(float(X_test_noisy[i, -1, meta['target_idx']]))
                    infer_total += (pred_end - pred_start)

                train_end = time.perf_counter()
                preds = np.asarray(preds, dtype=np.float64)
                t_train_s_per_ep = train_end - train_start
                t_inf_ms = infer_total / len(X_test_noisy) * 1000.0

            elif model_name == 'RandomForest':
                rf_n_estimators = 60
                rf_max_depth = 14
                rf_min_samples_leaf = 10
                rf_max_features = "sqrt"
                rf_random_state = 2024

                if pred_len >= 96:
                    max_train_samples = 8000
                else:
                    max_train_samples = 15000

                rng = np.random.default_rng(rf_random_state)
                if len(X_train) > max_train_samples:
                    sample_idx = rng.choice(len(X_train), size=max_train_samples, replace=False)
                    sample_idx = np.sort(sample_idx)
                    X_train_fit = X_train[sample_idx]
                    y_train_fit = y_train[sample_idx]
                else:
                    X_train_fit = X_train
                    y_train_fit = y_train

                logf.write(
                    "config: Fast RandomForestRegressor("
                    f"n_estimators={rf_n_estimators}, "
                    f"max_depth={rf_max_depth}, "
                    f"min_samples_leaf={rf_min_samples_leaf}, "
                    f"max_features={rf_max_features})\n"
                )
                logf.write(f"train_samples_original: {len(X_train)}\n")
                logf.write(f"train_samples_used: {len(X_train_fit)}\n")
                logf.write(f"test_samples_used: {len(X_test_noisy)}\n")
                logf.flush()

                train_start = time.perf_counter()
                base = RandomForestRegressor(
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                    min_samples_split=2,
                    min_samples_leaf=rf_min_samples_leaf,
                    max_features=rf_max_features,
                    bootstrap=True,
                    random_state=rf_random_state,
                    n_jobs=1
                )
                model = MultiOutputRegressor(estimator=base, n_jobs=4)
                model.fit(flatten_windows(X_train_fit), y_train_fit)
                train_end = time.perf_counter()

                infer_start = time.perf_counter()
                preds = model.predict(flatten_windows(X_test_noisy))
                infer_end = time.perf_counter()

                rf_tree_count = count_model_params(model)
                params_k = None
                t_train_s_per_ep = train_end - train_start
                t_inf_ms = (infer_end - infer_start) / len(X_test_noisy) * 1000.0
                logf.write(f"rf_tree_count: {rf_tree_count}\n")

            elif model_name == 'SVR':
                logf.write("config: Pipeline(StandardScaler + SVR(C=10.0, epsilon=0.01, kernel='rbf'))\n")
                train_start = time.perf_counter()
                base = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svr', SVR(C=10.0, epsilon=0.01, kernel='rbf'))
                ])
                model = MultiOutputRegressor(base)
                model.fit(flatten_windows(X_train), y_train)
                train_end = time.perf_counter()

                infer_start = time.perf_counter()
                preds = model.predict(flatten_windows(X_test_noisy))
                infer_end = time.perf_counter()

                t_train_s_per_ep = train_end - train_start
                t_inf_ms = (infer_end - infer_start) / len(X_test_noisy) * 1000.0

            elif model_name == 'Chronos':
                logf.write(
                    "config: ChronosPipeline.from_pretrained("
                    "amazon/chronos-t5-small, zero-shot, noisy historical power context, num_samples=20)\n"
                )
                logf.flush()

                train_start = time.perf_counter()
                train_end = train_start

                preds, params_k, mem_mb, t_inf_ms = run_chronos_forecast(
                    test_X=X_test_noisy,
                    pred_len=pred_len,
                    batch_size=64,
                    num_samples=20,
                )
                t_train_s_per_ep = 0.0

            else:
                raise ValueError(f'不支持的传统/外部模型: {model_name}')

            metrics = evaluate_and_save(
                result_dir=result_dir,
                model_name=model_name,
                dataset_name=dataset_name,
                pred_len=pred_len,
                preds=preds[..., None],
                trues=y_test_clean[..., None],
                params_k=params_k,
                mem_mb=mem_mb,
                t_train_s_per_ep=t_train_s_per_ep,
                t_inf_ms=t_inf_ms,
            )

            metrics['noise_level'] = float(noise_level)
            metrics['noise_include_target_history'] = bool(include_target_history)
            metrics['label_is_clean'] = True

            with open(result_dir / 'metrics.json', 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            save_txt_report(result_dir / 'run_report.txt', metrics)

            logf.write('\nfinal_metrics:\n')
            for k, v in metrics.items():
                logf.write(f'{k}: {v}\n')
            logf.flush()
            return metrics

        except Exception:
            logf.write('\n[ERROR]\n')
            logf.write(traceback.format_exc())
            logf.flush()
            raise

def run_repo_model(repo_root: str, run_py: str, model_repo_name: str, csv_meta: dict,
                   seq_len: int, label_len: int, pred_len: int, gpu: int,
                   result_root: Path, noise_level: float = 0.10, seed: int = 2024,
                   extra_args=None):
    extra_args = extra_args or []
    cmd = [
        sys.executable, run_py,
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--model', model_repo_name,
        '--data', 'custom',
        '--root_path', str(Path(csv_meta['csv_path']).parent),
        '--data_path', Path(csv_meta['csv_path']).name,
        '--features', 'MS',
        '--target', 'power',
        '--freq', 't',
        '--seq_len', str(seq_len),
        '--label_len', str(label_len),
        '--pred_len', str(pred_len),
        '--enc_in', str(csv_meta['enc_in']),
        '--dec_in', str(csv_meta['dec_in']),
        '--c_out', '1',
        '--inverse',
        '--gpu', '0',
        '--result_root', str(result_root),
        '--setting_suffix', csv_meta['dataset_name'],
        '--test_input_noise',
        '--test_noise_level', str(noise_level),
        '--test_noise_seed', str(seed),
        '--test_noise_include_target',
    ] + list(extra_args)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    tmp_log_dir = result_root / '_run_logs'
    tmp_log_dir.mkdir(parents=True, exist_ok=True)
    tmp_log_path = tmp_log_dir / f'{model_repo_name}__{csv_meta["dataset_name"]}__pl{pred_len}.log'

    print('RUN CMD:', ' '.join(cmd))
    tee_subprocess_to_log(
        cmd=cmd,
        cwd=repo_root,
        env=env,
        log_path=tmp_log_path
    )

    stem = Path(csv_meta['csv_path']).stem
    candidates = list(result_root.glob(f'{model_repo_name}__{stem}__*'))
    if not candidates:
        raise FileNotFoundError(f'未找到 {model_repo_name} 的结果目录，result_root={result_root}')
    latest = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    shutil.copy2(tmp_log_path, latest / 'train.log')

    with open(latest / 'metrics.json', 'r', encoding='utf-8') as f:
        return json.load(f), latest


def normalize_model_name(x: str) -> str:
    mapping = {
        'TimeLLM': 'Time-LLM',
        'TimeVLM': 'Time-VLM',
    }
    return mapping.get(x, x)
def normalize_dataset_name(x):
    if x is None:
        return None
    x = str(x)
    x_low = x.lower()
    if 'wind' in x_low:
        return 'wind'
    if x_low == 'pv' or x_low.startswith('pv') or 'pv_' in x_low:
        return 'PV'
    return None


def infer_dataset_from_item(item, result_dir: Path):
    for v in [
        item.get('dataset'),
        item.get('data_path'),
        item.get('setting'),
        result_dir.name,
    ]:
        ds = normalize_dataset_name(v)
        if ds is not None:
            return ds
    return None

def collect_summary(result_root: Path):
    rows = []
    for metrics_json in result_root.rglob('metrics.json'):
        with open(metrics_json, 'r', encoding='utf-8') as f:
            item = json.load(f)

        result_dir = metrics_json.parent
        item['result_dir'] = str(result_dir)

        if not item.get('model'):
            item['model'] = result_dir.name.split('__')[0]

        item['model'] = normalize_model_name(item['model'])

        # 关键：repo 模型的 metrics.json 通常没有 dataset 字段，
        # 需要从 data_path / setting / result_dir 里推断。
        item['dataset'] = infer_dataset_from_item(item, result_dir)

        rows.append(item)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    numeric_cols = [
        'pred_len', 'mae', 'sse', 'mse', 'rmse', 'mape', 'mspe', 'r2',
        'params_k', 'mem_mb', 't_train_s_per_ep', 't_inf_ms'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def compute_drop_table(clean_df: pd.DataFrame, noisy_df: pd.DataFrame):
    clean_df = clean_df.copy()
    noisy_df = noisy_df.copy()

    clean_df['model'] = clean_df['model'].map(normalize_model_name)
    noisy_df['model'] = noisy_df['model'].map(normalize_model_name)

    clean_df['dataset'] = clean_df['dataset'].map(normalize_dataset_name)
    noisy_df['dataset'] = noisy_df['dataset'].map(normalize_dataset_name)

    merged = pd.merge(
        clean_df[['model', 'dataset', 'pred_len', 'mae', 'mse']].rename(
            columns={'mae': 'mae_clean', 'mse': 'mse_clean'}
        ),
        noisy_df[['model', 'dataset', 'pred_len', 'mae', 'mse']].rename(
            columns={'mae': 'mae_noisy', 'mse': 'mse_noisy'}
        ),
        on=['model', 'dataset', 'pred_len'],
        how='inner'
    )

    merged['mae_drop_pct'] = (
        merged['mae_noisy'] - merged['mae_clean']
    ) / merged['mae_clean'] * 100.0

    merged['mse_drop_pct'] = (
        merged['mse_noisy'] - merged['mse_clean']
    ) / merged['mse_clean'] * 100.0

    return merged


def main():
    parser = argparse.ArgumentParser(description='Robustness test with 10% Gaussian white noise on test set (T=1)')
    parser.add_argument('--repo_root', type=str, required=True)
    parser.add_argument('--run_py', type=str, default='run.py')
    parser.add_argument('--wind_path', type=str, default='')
    parser.add_argument('--pv_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='both', choices=['both', 'wind', 'PV'],
                        help='选择只跑哪个数据集：both / wind / PV')
    parser.add_argument('--clean_summary_path', type=str, required=True,
                        help='已有干净测试结果的 summary_accuracy.csv 路径')
    parser.add_argument('--out_root', type=str, default='./benchmark_noise_t1_gwn10')
    parser.add_argument('--seq_len', type=int, default=288)
    parser.add_argument('--label_len', type=int, default=144)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--noise_level', type=float, default=0.10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_root = Path(args.out_root)
    prepared_root = out_root / 'prepared_data'

    result_root = out_root / 'benchmark_results'
    result_root.mkdir(parents=True, exist_ok=True)

    # 1) 根据参数选择数据集。这里不再生成 noisy CSV。
    # 噪声在窗口级别或 test loop 中加入，保证 y_true 始终是 clean label。
    dataset_metas = []

    if args.dataset in ['both', 'wind']:
        if not args.wind_path:
            raise ValueError("选择 wind 或 both 时必须提供 --wind_path")

        wind_clean = standardize_energy_file(args.wind_path, prepared_root)
        wind_clean['dataset_name'] = 'wind'
        dataset_metas.append(wind_clean)

    if args.dataset in ['both', 'PV']:
        if not args.pv_path:
            raise ValueError("选择 PV 或 both 时必须提供 --pv_path")

        pv_clean = standardize_energy_file(args.pv_path, prepared_root)
        pv_clean['dataset_name'] = 'PV'
        dataset_metas.append(pv_clean)

    classical_models = ['ARIMA', 'ETS', 'RandomForest', 'SVR', 'Chronos']

    repo_display_models = [
        'LSTM',
        'GRU',
        'DLinear',
        'TimesNet',
        'Informer',
        'PatchTST',
        'iTransformer',
        'Time-LLM',
        'Time-VLM',
    ]
    display_to_repo = {
        'LSTM': 'LSTM',
        'TCN': 'TCN',
        'DLinear': 'DLinear',
        'TimesNet': 'TimesNet',
        'Informer': 'Informer',
        'PatchTST': 'PatchTST',
        'iTransformer': 'iTransformer',
        'Time-LLM': 'TimeLLM',
        'Time-VLM': 'TimeVLM',
        'GRU': 'GRU',
    }

    # 3) 跑 noisy test 实验（T=1）
    for meta in dataset_metas:
        print(f'\n===== ROBUSTNESS | DATASET={meta["dataset_name"]} | pred_len={args.pred_len} | noise={args.noise_level:.2f} =====')

        for m in classical_models:
            try:
                run_classical_baseline(
                    model_name=m,
                    csv_path=meta['csv_path'],
                    seq_len=args.seq_len,
                    pred_len=args.pred_len,
                    result_root=result_root,
                    noise_level=args.noise_level,
                    seed=args.seed,
                    include_target_history=True,
                )
                print(f'[OK] classical {m} on {meta["dataset_name"]}')
            except Exception as e:
                print(f'[FAIL] classical {m} on {meta["dataset_name"]}: {e}')

        for disp_name in repo_display_models:
            repo_name = display_to_repo[disp_name]
            try:
                run_repo_model(
                    repo_root=args.repo_root,
                    run_py=args.run_py,
                    model_repo_name=repo_name,
                    csv_meta=meta,
                    seq_len=args.seq_len,
                    label_len=args.label_len,
                    pred_len=args.pred_len,
                    gpu=args.gpu,
                    result_root=result_root,
                    noise_level=args.noise_level,
                    seed=args.seed,
                )
                print(f'[OK] repo model {disp_name} on {meta["dataset_name"]}')
            except Exception as e:
                print(f'[FAIL] repo model {disp_name} on {meta["dataset_name"]}: {e}')

    # 4) 收集 noisy 结果
    noisy_df = collect_summary(result_root)
    noisy_df.to_csv(out_root / 'summary_noisy_results.csv', index=False, encoding='utf-8-sig')

    # 5) 读取 clean summary_accuracy.csv，只保留 pred_len=1
    clean_df = pd.read_csv(args.clean_summary_path)
    clean_df['model'] = clean_df['model'].map(normalize_model_name)
    clean_df = clean_df[clean_df['pred_len'] == args.pred_len].copy()

    # 6) 计算性能下降百分比
    drop_df = compute_drop_table(clean_df, noisy_df)
    drop_df.to_csv(out_root / 'robustness_drop_t1.csv', index=False, encoding='utf-8-sig')

    print('\n结果已保存到：')
    print(out_root)
    print(out_root / 'summary_noisy_results.csv')
    print(out_root / 'robustness_drop_t1.csv')


if __name__ == '__main__':
    main()