# Model cost functions for Gonzalez et. al. 2020
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/bryangonzalez/Documents/Dartmouth/Research/TheoryOfMind_Bx_experiment/Code/models/')
import comp_models



def rw_costfun(alpha, game):
    model_predictions = pd.Series(comp_models.rw_learn(game, alpha))
    observations = game['ret'].reset_index(drop=True)
    residuals = model_predictions - observations
    return residuals.astype('float')


def imm_costfun(params, game, folk_thry):
    alpha = params[0]
    tau = params[1]
    model_predictions = pd.Series(comp_models.imm_learn(game, folk_thry, alpha, tau))
    observations = game['ret'].reset_index(drop=True)
    residuals = model_predictions - observations
    return residuals.astype('float')


def gr_costfun(game):
    model_predictions = pd.Series([comp_models.gr_model(game['inv'][trl], game['mult'][trl], theta=0, phi=0)
                                   for trl in np.arange(len(game))])
    observations = game['ret']
    residuals = model_predictions - observations
    return residuals.tolist()


def ga_costfun(theta, game):
    model_predictions = pd.Series([comp_models.ga_model(game['inv'][trl], game['mult'][trl], theta=theta, phi=0)
                                   for trl in np.arange(len(game))])
    observations = game['ret']
    residuals = model_predictions - observations
    return residuals.tolist()


def ia_costfun(theta, game):
    model_predictions = pd.Series([comp_models.ia_model(game['inv'][trl], game['mult'][trl], theta=theta, phi=0)
                                   for trl in np.arange(len(game))])
    observations = game['ret']
    residuals = model_predictions - observations
    return residuals.tolist()


def mp_costfun(param, game):
    thayta = param[0]
    phee = param[1]
    model_predictions = pd.Series([comp_models.mp_model_ppsoe(game['inv'][trl], game['mult'][trl], theta=thayta, phi=phee)
                                   for trl in np.arange(len(game))])
    observations = game['ret']
    residuals = model_predictions - observations
    return residuals.tolist()



