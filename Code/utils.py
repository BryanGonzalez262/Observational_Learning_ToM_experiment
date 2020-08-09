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


def soft_max(x, tau):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(np.array(x) / tau)
    return e_x / e_x.sum(axis=0)


def old_softmax(x, temperature):
    EPSILON = 10e-16  # to avoid taking the log of zero
    # print(preds)
    x = (np.array(x+ EPSILON)).astype('float64')
    preds = np.log(x) / temperature
    # print(preds)
    exp_preds = np.exp(preds)
    # print(exp_preds)
    preds = exp_preds / np.sum(exp_preds)
    # print(preds)
    probas = np.random.multinomial(1, preds, 1)
    return probas[0]


def softmax2(X, tau = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(tau)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p