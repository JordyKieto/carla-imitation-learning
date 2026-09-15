"""Offline action-prediction metrics logged during training (names match results/barzksni/history.csv).

All inputs are numpy arrays: targets/preds (B, 3) as [throttle, brake, steer],
directions (B,) as ids into DIRECTION_NAMES.
"""
import numpy as np

from dataset.records import CONTROL_NAMES
from dataset.windows import DIRECTION_NAMES

TURN_THRESHOLD = 0.1
LAG_STEPS = (1, 2, 3)
ROLLOUT_LEN = 10
EPS = 1e-8


def control_metrics(targets, preds):
    err = preds - targets
    mse = np.mean(err ** 2, axis=0)
    target_mean, pred_mean = targets.mean(axis=0), preds.mean(axis=0)
    target_var, pred_var = targets.var(axis=0), preds.var(axis=0)
    ss_tot = np.sum((targets - target_mean) ** 2, axis=0)
    metrics = {}
    for i, name in enumerate(CONTROL_NAMES):
        metrics[f"mae/{name}"] = np.mean(np.abs(err[:, i]))
        metrics[f"rmse/{name}"] = np.sqrt(mse[i] + EPS)
        metrics[f"r2/{name}"] = 1.0 - np.sum(err[:, i] ** 2) / (ss_tot[i] + EPS)
        metrics[f"bias/{name}"] = pred_mean[i] - target_mean[i]
        metrics[f"dist/mean_diff_{name}"] = pred_mean[i] - target_mean[i]
        metrics[f"dist/var_ratio_{name}"] = pred_var[i] / (target_var[i] + EPS)
        metrics[f"uncertainty/pred_var_{name}"] = pred_var[i]
    return metrics


def temporal_metrics(targets, preds):
    n = len(preds)
    if n < 2:
        return {}
    metrics = {}
    smoothness = np.mean(np.abs(np.diff(preds, axis=0)), axis=0)
    for i, name in enumerate(CONTROL_NAMES):
        metrics[f"smoothness/{name}"] = smoothness[i]
    for k in LAG_STEPS:
        if n > k:
            lag = np.mean(np.abs(preds[:-k] - targets[k:]), axis=0)
            for i, name in enumerate(CONTROL_NAMES):
                metrics[f"lag/{name}_k{k}"] = lag[i]

    pred_turn = np.abs(preds[:, 2]) > TURN_THRESHOLD
    target_turn = np.abs(targets[:, 2]) > TURN_THRESHOLD
    tp = np.sum(pred_turn & target_turn)
    precision = tp / (pred_turn.sum() + EPS)
    recall = tp / (target_turn.sum() + EPS)
    metrics["turn/precision"] = precision
    metrics["turn/recall"] = recall
    metrics["turn/f1"] = 2 * precision * recall / (precision + recall + EPS)
    delay_valid = pred_turn.any() and target_turn.any()
    metrics["turn/delay_steps"] = float(np.argmax(pred_turn) - np.argmax(target_turn)) if delay_valid else 0.0
    metrics["turn/delay_valid"] = float(delay_valid)

    err_mag = np.mean(np.abs(preds - targets), axis=1)
    pred_disp = np.mean((preds - preds.mean(axis=0)) ** 2, axis=1)
    ec, dc = err_mag - err_mag.mean(), pred_disp - pred_disp.mean()
    metrics["uncertainty/error_variance_corr"] = np.sum(ec * dc) / (np.sqrt(np.sum(ec ** 2) * np.sum(dc ** 2)) + EPS)

    h = min(ROLLOUT_LEN, n)
    rollout = np.abs(preds[:h] - targets[:h])
    metrics[f"rollout/control_error_h{h}"] = rollout.mean()
    metrics[f"rollout/steer_error_h{h}"] = rollout[:, 2].mean()
    return metrics


def direction_losses(targets, preds, directions):
    """Returns {direction: (mean per-sample mse, count)}."""
    per_sample = np.mean((targets - preds) ** 2, axis=1)
    out = {}
    for idx, name in enumerate(DIRECTION_NAMES):
        mask = directions == idx
        out[name] = (float(per_sample[mask].mean()) if mask.any() else 0.0, int(mask.sum()))
    return out


def batch_metrics(targets, preds):
    metrics = control_metrics(targets, preds)
    metrics.update(temporal_metrics(targets, preds))
    return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in metrics.items()}
