import numpy as np
import pandas as pd

def brier_score(predictions, outcomes):
    predictions = np.array(predictions)
    outcomes = np.array(outcomes)
    return np.mean((predictions - outcomes) ** 2)

def calibration_curve(predictions, outcomes, n_bins=10):
    predictions = np.array(predictions)
    outcomes = np.array(outcomes)
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(predictions, bins) - 1
    bin_sums = np.zeros(n_bins)
    bin_total = np.zeros(n_bins)
    for i, b in enumerate(binids):
        if 0 <= b < n_bins:
            bin_sums[b] += outcomes[i]
            bin_total[b] += 1
    prob_true = np.divide(bin_sums, bin_total, out=np.zeros_like(bin_sums), where=bin_total!=0)
    prob_pred = (bins[:-1] + bins[1:]) / 2
    return prob_pred, prob_true, bin_total

def peer_score(forecaster_scores, all_scores):
    mean_scores = np.mean(all_scores, axis=0)
    return np.mean(forecaster_scores - mean_scores)

def coverage(pred_intervals, outcomes):
    covered = [lower <= outcome <= upper for (lower, upper), outcome in zip(pred_intervals, outcomes)]
    return np.mean(covered)

def brier_score_df(df, pred_col='prediction', outcome_col='outcome'):
    return brier_score(df[pred_col], df[outcome_col])

def calibration_curve_df(df, pred_col='prediction', outcome_col='outcome', n_bins=10):
    return calibration_curve(df[pred_col], df[outcome_col], n_bins=n_bins)

def coverage_df(df, lower_col='lower', upper_col='upper', outcome_col='outcome'):
    pred_intervals = list(zip(df[lower_col], df[upper_col]))
    return coverage(pred_intervals, df[outcome_col]) 