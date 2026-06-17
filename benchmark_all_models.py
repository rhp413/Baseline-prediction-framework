import argparse
import json

import subprocess
import sys
import time
from pathlib import Path
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import shutil
import traceback
from utils.dm_test import dm_test
from utils.metrics import metric_extended
try:
    from chronos import ChronosPipeline
    HAS_CHRONOS = True
except Exception:
    HAS_CHRONOS = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]+', '_', s).strip('_')


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
    raise ValueError(f'未检测到时间列，当前列名：{columns}')


def detect_target_col(columns):
    for c in columns:
        lc = str(c).lower()
        if 'power' in lc:
            return c
    raise ValueError(f'未检测到目标列(power)，当前列名：{columns}')


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
        rename_map[c] = safe_name(c.lower())

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
        'time_col_original': str(time_col),
        'target_col_original': str(target_col),
        'enc_in': len(df.columns) - 1,
        'dec_in': len(df.columns) - 1,
        'c_out': 1,
        'n_total_features_without_date': len(df.columns) - 1,
        'n_exogenous_without_target': len(df.columns) - 2,
    }


def build_windows_from_standard_csv(csv_path: str, seq_len: int, pred_len: int):
    df = pd.read_csv(csv_path)
    assert 'date' in df.columns and 'power' in df.columns

    values = df.drop(columns=['date']).values.astype(np.float64)
    target_idx = list(df.drop(columns=['date']).columns).index('power')

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

    return splits


def save_txt_report(path: str, info: dict):
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in info.items():
            f.write(f'{k}: {v}\n')
def tee_subprocess_to_log(cmd, cwd, env, log_path: Path):
    """
    将子进程 stdout/stderr 同时打印到终端并保存到日志文件
    """
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
            print(line, end='')     # 终端照常显示
            lf.write(line)          # 同时写文件
            lf.flush()

        return_code = proc.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

def run_arima(train_series, test_X, pred_len):
    if not HAS_STATSMODELS:
        raise RuntimeError('statsmodels 未安装，无法运行 ARIMA。')

    history = list(train_series.astype(float))
    preds = []
    for i in range(len(test_X)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=pred_len)
        preds.append(np.asarray(forecast, dtype=np.float64))
        history.append(float(test_X[i, -1, -1]))
    return np.asarray(preds, dtype=np.float64)


def run_ets(train_series, test_X, pred_len, seasonal_periods=96):
    if not HAS_STATSMODELS:
        raise RuntimeError('statsmodels 未安装，无法运行 ETS。')

    history = list(train_series.astype(float))
    preds = []
    for i in range(len(test_X)):
        try:
            model = ExponentialSmoothing(
                history,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods,
                initialization_method='estimated'
            )
            model_fit = model.fit(optimized=True, use_brute=False)
            forecast = model_fit.forecast(pred_len)
        except Exception:
            model = ExponentialSmoothing(
                history,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
            model_fit = model.fit(optimized=True, use_brute=False)
            forecast = model_fit.forecast(pred_len)

        preds.append(np.asarray(forecast, dtype=np.float64))
        history.append(float(test_X[i, -1, -1]))
    return np.asarray(preds, dtype=np.float64)


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

def run_svr(train_X, train_y, test_X):
    base = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(C=10.0, epsilon=0.01, kernel='rbf'))
    ])
    model = MultiOutputRegressor(base)
    model.fit(flatten_windows(train_X), train_y)
    pred = model.predict(flatten_windows(test_X))
    return pred, model


def count_model_params(model) -> float:
    if model is None:
        return None

    total = 0

    if hasattr(model, 'estimators_'):
        for est in model.estimators_:
            if hasattr(est, 'get_booster'):
                booster = est.get_booster()
                total += len(booster.get_dump())

            # RandomForestRegressor
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


def run_classical_baseline(model_name: str, csv_path: str, seq_len: int, pred_len: int, result_root: Path):
    splits = build_windows_from_standard_csv(csv_path, seq_len=seq_len, pred_len=pred_len)
    X_train, y_train = splits['train']
    X_test, y_test = splits['test']

    train_series = X_train[:, -1, -1]
    dataset_name = Path(csv_path).stem.replace('_std', '')
    result_dir = result_root / f'{model_name}__{dataset_name}__pl{pred_len}'
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
        logf.write(f'X_train.shape: {X_train.shape}\n')
        logf.write(f'y_train.shape: {y_train.shape}\n')
        logf.write(f'X_test.shape: {X_test.shape}\n')
        logf.write(f'y_test.shape: {y_test.shape}\n\n')
        logf.flush()

        try:
            if model_name == 'ARIMA':
                logf.write('config: ARIMA(order=(5,1,0))\n')
                train_start = time.perf_counter()
                preds = []
                history = list(train_series.astype(float))
                infer_total = 0.0
                for i in range(len(X_test)):
                    model = ARIMA(history, order=(5, 1, 0))
                    model_fit = model.fit()

                    pred_start = time.perf_counter()
                    forecast = model_fit.forecast(steps=pred_len)
                    pred_end = time.perf_counter()

                    preds.append(np.asarray(forecast, dtype=np.float64))
                    history.append(float(X_test[i, -1, -1]))
                    infer_total += (pred_end - pred_start)
                train_end = time.perf_counter()

                preds = np.asarray(preds, dtype=np.float64)
                t_train_s_per_ep = train_end - train_start
                t_inf_ms = infer_total / len(X_test) * 1000.0

            elif model_name == 'ETS':
                logf.write("config: ExponentialSmoothing(trend='add', seasonal='add', seasonal_periods=96)\n")
                train_start = time.perf_counter()
                preds = []
                history = list(train_series.astype(float))
                infer_total = 0.0
                for i in range(len(X_test)):
                    try:
                        model = ExponentialSmoothing(
                            history, trend='add', seasonal='add',
                            seasonal_periods=96, initialization_method='estimated'
                        )
                        model_fit = model.fit(optimized=True, use_brute=False)
                    except Exception as e:
                        logf.write(f'ETS seasonal fit failed at step {i}: {e}\n')
                        model = ExponentialSmoothing(
                            history, trend='add', seasonal=None,
                            initialization_method='estimated'
                        )
                        model_fit = model.fit(optimized=True, use_brute=False)

                    pred_start = time.perf_counter()
                    forecast = model_fit.forecast(pred_len)
                    pred_end = time.perf_counter()

                    preds.append(np.asarray(forecast, dtype=np.float64))
                    history.append(float(X_test[i, -1, -1]))
                    infer_total += (pred_end - pred_start)
                train_end = time.perf_counter()

                preds = np.asarray(preds, dtype=np.float64)
                t_train_s_per_ep = train_end - train_start
                t_inf_ms = infer_total / len(X_test) * 1000.0

            elif model_name == 'RandomForest':
                """
                Fast RandomForest baseline.

                说明：
                1. T=96 时不再使用全部 5 万多个训练窗口，而是随机采样部分窗口。
                2. 限制树深度和树数量，避免完整数据集下训练时间过长。
                3. 测试集仍然完整测试，不做子采样。
                """

                rf_n_estimators = 80
                rf_max_depth = 16
                rf_min_samples_leaf = 5
                rf_max_features = "sqrt"
                rf_random_state = 2024

                # T=96 极慢，所以采样更激进；T=1 可以稍微多用一些样本
                if pred_len >= 96:
                    max_train_samples = 12000
                else:
                    max_train_samples = 25000

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
                    f"max_features={rf_max_features}, "
                    f"random_state={rf_random_state})\n"
                )
                logf.write(f"train_samples_original: {len(X_train)}\n")
                logf.write(f"train_samples_used: {len(X_train_fit)}\n")
                logf.write(f"test_samples_used: {len(X_test)}\n")
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

                # 这里让 MultiOutputRegressor 并行不同预测步长，避免和 RF 内部 n_jobs 嵌套爆内存
                model = MultiOutputRegressor(
                    estimator=base,
                    n_jobs=4
                )

                model.fit(flatten_windows(X_train_fit), y_train_fit)
                train_end = time.perf_counter()

                infer_start = time.perf_counter()
                preds = model.predict(flatten_windows(X_test))
                infer_end = time.perf_counter()

                rf_tree_count = count_model_params(model)
                params_k = None

                t_train_s_per_ep = train_end - train_start
                t_inf_ms = (infer_end - infer_start) / len(X_test) * 1000.0

                logf.write(f"rf_tree_count: {rf_tree_count}\n")
                logf.write(f"t_train_s_per_ep: {t_train_s_per_ep}\n")
                logf.write(f"t_inf_ms: {t_inf_ms}\n")
                logf.flush()
            elif model_name == 'Chronos':
                logf.write(
                    "config: ChronosPipeline.from_pretrained("
                    "amazon/chronos-t5-small, zero-shot, target-only power context, "
                    "num_samples=20)\n"
                )
                logf.write(
                    "note: Chronos-T5 is used as a zero-shot univariate model. "
                    "Only historical power values are used as context; exogenous features are ignored.\n"
                )
                logf.flush()

                train_start = time.perf_counter()
                # Chronos 不做任务特定训练，因此训练时间记为 0
                train_end = train_start

                preds, params_k, mem_mb, t_inf_ms = run_chronos_forecast(
                    test_X=X_test,
                    pred_len=pred_len,
                    batch_size=64,
                    num_samples=20,
                )

                t_train_s_per_ep = 0.0

                logf.write(f'params_k: {params_k}\n')
                logf.write(f'mem_mb: {mem_mb}\n')
                logf.write(f't_train_s_per_ep: {t_train_s_per_ep}\n')
                logf.write(f't_inf_ms: {t_inf_ms}\n')
                logf.flush()
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
                preds = model.predict(flatten_windows(X_test))
                infer_end = time.perf_counter()

                t_train_s_per_ep = train_end - train_start
                t_inf_ms = (infer_end - infer_start) / len(X_test) * 1000.0

            else:
                raise ValueError(f'不支持的传统模型: {model_name}')

            metrics = evaluate_and_save(
                result_dir=result_dir,
                model_name=model_name,
                dataset_name=dataset_name,
                pred_len=pred_len,
                preds=preds[..., None],
                trues=y_test[..., None],
                params_k=params_k,
                mem_mb=mem_mb,
                t_train_s_per_ep=t_train_s_per_ep,
                t_inf_ms=t_inf_ms,
            )

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
                   result_root: Path, extra_args=None):
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
    ] + list(extra_args)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # 先写一个临时日志，保证即使失败也能保留
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

    # 把完整训练日志复制到该实验目录
    shutil.copy2(tmp_log_path, latest / 'train.log')

    with open(latest / 'metrics.json', 'r', encoding='utf-8') as f:
        return json.load(f), latest


def _normalize_dataset_name(x):
    if x is None:
        return None
    x = str(x).strip()
    if not x:
        return None

    x_low = x.lower()

    # 直接匹配常见名字
    if 'wind' in x_low:
        return 'wind'
    if x_low == 'pv' or 'pv_' in x_low or x_low.startswith('pv'):
        return 'PV'

    # 去掉后缀后再判断
    x_low = x_low.replace('.csv', '').replace('.xlsx', '')
    x_low = x_low.replace('_std', '').replace('_standardized', '')
    if 'wind' in x_low:
        return 'wind'
    if x_low == 'pv' or x_low.startswith('pv'):
        return 'PV'

    return None


def _infer_dataset_from_item(item, result_dir: Path):
    candidates = [
        item.get('dataset'),
        item.get('data_path'),
        item.get('setting'),
        result_dir.name,
    ]
    for c in candidates:
        ds = _normalize_dataset_name(c)
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

        # 1) 回填 dataset，修复 repo 模型 dataset 为空的问题
        item['dataset'] = _infer_dataset_from_item(item, result_dir)

        # 2) 如果某些结果里 model 缺失，可从目录名补
        if not item.get('model'):
            item['model'] = result_dir.name.split('__')[0]

        # 3) pred_len 尽量转成 int，避免后面分组混乱
        if 'pred_len' in item and item['pred_len'] is not None:
            try:
                item['pred_len'] = int(item['pred_len'])
            except Exception:
                pass

        rows.append(item)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 统一数值列类型
    numeric_cols = [
        'pred_len', 'mae', 'sse', 'mse', 'rmse', 'mape', 'mspe', 'r2',
        'params_k', 'mem_mb', 't_train_s_per_ep', 't_inf_ms', 'dtw'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def compute_dm_table(result_root: Path, anchor_display_name: str, display_to_repo: dict):
    summary = collect_summary(result_root)
    if summary.empty:
        return pd.DataFrame()

    out_rows = []
    for dataset in sorted(summary['dataset'].dropna().unique()):
        for pred_len in sorted(summary['pred_len'].dropna().unique()):
            sub = summary[(summary['dataset'] == dataset) & (summary['pred_len'] == pred_len)]
            anchor_row = sub[sub['model'] == display_to_repo.get(anchor_display_name, anchor_display_name)]
            if anchor_row.empty:
                anchor_row = sub[sub['model'] == anchor_display_name]
            if anchor_row.empty:
                continue

            anchor_dir = Path(anchor_row.iloc[0]['result_dir'])
            anchor_pred = np.load(anchor_dir / 'pred.npy')
            anchor_true = np.load(anchor_dir / 'true.npy')

            for _, row in sub.iterrows():
                model_name = row['model']
                model_dir = Path(row['result_dir'])
                pred = np.load(model_dir / 'pred.npy')
                true = np.load(model_dir / 'true.npy')

                if pred.shape != anchor_pred.shape or true.shape != anchor_true.shape:
                    continue

                dm_stat, p_value = dm_test(
                    actual=anchor_true.reshape(-1),
                    pred1=anchor_pred.reshape(-1),
                    pred2=pred.reshape(-1),
                    h=1,
                    loss_type='mse',
                )
                out_rows.append({
                    'dataset': dataset,
                    'pred_len': int(pred_len),
                    'anchor_model': anchor_display_name,
                    'compare_model': model_name,
                    'dm_stat': dm_stat,
                    'p_value': p_value,
                    'significant_p_lt_0_05': bool(p_value < 0.05) if pd.notna(p_value) else None,
                })
    return pd.DataFrame(out_rows)


def main():
    parser = argparse.ArgumentParser(description='Wind/PV baseline benchmark orchestrator')
    parser.add_argument('--repo_root', type=str, required=True, help='你的仓库根目录')
    parser.add_argument('--run_py', type=str, default='run.py')
    parser.add_argument('--wind_path', type=str, required=True)
    parser.add_argument('--pv_path', type=str, required=True)
    parser.add_argument('--out_root', type=str, default='./benchmark_master')
    parser.add_argument('--seq_len', type=int, default=288)
    parser.add_argument('--label_len', type=int, default=144)
    parser.add_argument('--pred_lens', type=int, nargs='+', default=[1, 16, 96])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--anchor_model', type=str, default='Time-VLM')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_root = Path(args.out_root)
    data_root = out_root / 'prepared_data'
    result_root = out_root / 'benchmark_results'
    result_root.mkdir(parents=True, exist_ok=True)

    wind_meta = standardize_energy_file(args.wind_path, data_root)
    pv_meta = standardize_energy_file(args.pv_path, data_root)

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

    classical_models = ['ARIMA', 'ETS', 'RandomForest', 'SVR', 'Chronos']
    repo_display_models = ['LSTM', 'GRU', 'DLinear', 'TimesNet', 'Informer', 'PatchTST', 'iTransformer', 'Time-LLM', 'Time-VLM']

    dataset_metas = [wind_meta, pv_meta]

    for meta in dataset_metas:
        for pred_len in args.pred_lens:
            print(f'\n===== DATASET={meta["dataset_name"]}, pred_len={pred_len} =====')

            for m in classical_models:
                try:
                    run_classical_baseline(
                        model_name=m,
                        csv_path=meta['csv_path'],
                        seq_len=args.seq_len,
                        pred_len=pred_len,
                        result_root=result_root,
                    )
                    print(f'[OK] classical {m} on {meta["dataset_name"]}, pred_len={pred_len}')
                except Exception as e:
                    print(f'[FAIL] classical {m} on {meta["dataset_name"]}, pred_len={pred_len}: {e}')

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
                        pred_len=pred_len,
                        gpu=args.gpu,
                        result_root=result_root,
                    )
                    print(f'[OK] repo model {disp_name} on {meta["dataset_name"]}, pred_len={pred_len}')
                except Exception as e:
                    print(f'[FAIL] repo model {disp_name} on {meta["dataset_name"]}, pred_len={pred_len}: {e}')

    summary = collect_summary(result_root)
    if not summary.empty:
        summary.to_csv(out_root / 'summary_all_results.csv', index=False, encoding='utf-8-sig')

        accuracy_cols = ['model', 'dataset', 'pred_len', 'mae', 'sse', 'mse', 'rmse', 'r2']
        efficiency_cols = ['model', 'dataset', 'pred_len', 'params_k', 'mem_mb', 't_train_s_per_ep', 't_inf_ms']

        summary[accuracy_cols].to_csv(out_root / 'summary_accuracy.csv', index=False, encoding='utf-8-sig')
        summary[efficiency_cols].to_csv(out_root / 'summary_efficiency.csv', index=False, encoding='utf-8-sig')

        dm_df = compute_dm_table(result_root, anchor_display_name=args.anchor_model, display_to_repo=display_to_repo)
        dm_df.to_csv(out_root / 'summary_dm_test.csv', index=False, encoding='utf-8-sig')

        print(f'\n汇总文件已生成到: {out_root}')
        print(out_root / 'summary_all_results.csv')
        print(out_root / 'summary_accuracy.csv')
        print(out_root / 'summary_efficiency.csv')
        print(out_root / 'summary_dm_test.csv')
    else:
        print('没有收集到任何结果，请检查模型名、路径和依赖。')


if __name__ == '__main__':
    main()
