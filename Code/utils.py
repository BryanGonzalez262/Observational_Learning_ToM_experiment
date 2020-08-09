# Imports
import numpy as np
import pandas as pd


folk_params = pd.DataFrame({'theta': [.5, 0, 0, .13], 'phi': [0, -.1, .1, 0]}, index=['gr', 'ga', 'ia', 'mo'])


def compute_bic(n_obs, sse, n_par):
    bic = n_obs * np.log(sse / n_obs) + n_par * np.log(n_obs)
    return bic


def compute_aic(n_obs, sse, n_par):
    aic = n_obs * np.log(sse/n_obs) + 2*n_par
    return aic


def old_soft_max(x, tau):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(np.array(x) / tau)
    return e_x / e_x.sum(axis=0)


def soft_max(x, tau):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(np.divide(np.array(x), tau))
    r = np.divide(e_x, e_x.sum(axis=0))
    if np.isnan(r).any():
        xx = x - np.max(x)
        e_xx = np.exp(np.divide(np.array(xx), tau))
        r = np.divide(e_xx, e_xx.sum(axis=0))
    return r