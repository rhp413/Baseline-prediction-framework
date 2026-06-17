import numpy as np
import scipy.stats as stats


def dm_test(actual, pred1, pred2, h=1, loss_type='mse'):
    """
    Diebold-Mariano 检验。

    参数
    ----
    actual : array-like
        真实值。
    pred1 : array-like
        模型1预测值。建议传“你的模型 / 作为比较基准的优模型”。
    pred2 : array-like
        模型2预测值。
    h : int
        预测步长。表格整体比较时，通常可取 1；如要严格按 pred_len 修正，可传 pred_len。
    loss_type : str
        'mse' 或 'mae'

    返回
    ----
    dm_stat : float
    p_value : float
    """
    actual = np.asarray(actual).reshape(-1)
    pred1 = np.asarray(pred1).reshape(-1)
    pred2 = np.asarray(pred2).reshape(-1)

    if not (len(actual) == len(pred1) == len(pred2)):
        raise ValueError("actual, pred1, pred2 长度必须一致。")

    e1 = actual - pred1
    e2 = actual - pred2

    if loss_type.lower() == 'mse':
        d = (e1 ** 2) - (e2 ** 2)
    elif loss_type.lower() == 'mae':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss_type 仅支持 'mse' 或 'mae'。")

    d_mean = np.mean(d)
    T = len(d)

    if T <= 1:
        return np.nan, np.nan

    h = max(1, min(int(h), T - 1))

    gamma = []
    for lag in range(h):
        v1 = d[lag:] - d_mean
        v2 = d[:T - lag] - d_mean
        gamma.append(np.mean(v1 * v2))

    var_d = gamma[0] + 2.0 * np.sum(gamma[1:])

    if var_d <= 0 or np.isnan(var_d):
        return np.nan, np.nan

    dm_stat = d_mean / np.sqrt(var_d / T)
    p_value = 2 * stats.norm.cdf(-abs(dm_stat))
    return float(dm_stat), float(p_value)