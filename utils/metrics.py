import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def SSE(pred, true):
    return np.sum((true - pred) ** 2)


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true, eps=1e-8):
    true_safe = np.where(np.abs(true) < eps, eps, true)
    return np.mean(np.abs((true - pred) / true_safe))


def MSPE(pred, true, eps=1e-8):
    true_safe = np.where(np.abs(true) < eps, eps, true)
    return np.mean(np.square((true - pred) / true_safe))


def R2(pred, true):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)

    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)

    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return mae, mse, rmse, mape, mspe


def metric_extended(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)

    mae = MAE(pred, true)
    sse = SSE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = R2(pred, true)

    return mae, sse, mse, rmse, mape, mspe, r2