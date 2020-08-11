# Computational models used in Gonzalez et al 2020

import numpy as np
import sys
sys.path.append('/Users/bryangonzalez/Documents/Dartmouth/Research/TheoryOfMind_Bx_experiment/Code/')
import utils

#%% Modified Rescorla-Wagner (model-free RL)


def rw_learn(game, alpha):
    num_trls = game.shape[0]
    predicted_bx = np.zeros(num_trls)
    v_2 = np.zeros(num_trls + 1)
    v_4 = np.zeros(num_trls + 1)
    v_6 = np.zeros(num_trls + 1)
    for t in np.arange(num_trls):
        if game.mult.iloc[t] == 2:
            v_2[t + 1] = v_2[t] + alpha * (game.ret.iloc[t] - v_2[t])
            v_4[t + 1] = v_4[t]
            v_6[t + 1] = v_6[t]
            predicted_bx[t] = v_2[t]

        if game.mult.iloc[t] == 4:
            v_4[t + 1] = v_4[t] + alpha * (game.ret.iloc[t] - v_4[t])
            v_2[t + 1] = v_2[t]
            v_6[t + 1] = v_6[t]
            predicted_bx[t] = v_4[t]

        if game.mult.iloc[t] == 6:
            v_6[t + 1] = v_6[t] + alpha * (game.ret.iloc[t] - v_6[t])
            v_2[t + 1] = v_2[t]
            v_4[t + 1] = v_4[t]
            predicted_bx[t] = v_6[t]
    predicted_bx[np.isnan(predicted_bx)] = 0
    return predicted_bx

#%% Inverse Motivation Model


def imm_learn(game, folk_thry, alpha, tau):
    strat_type = ['gr', 'ga', 'ia', 'mo']  # The agent's heuristic 'state space'
    p_strat = np.zeros([len(game)+1, 4])  # store the agent's belief of the prob that the trustee is using each strategy
    predicted_bx = np.zeros([len(game)+1, 4]) #counterfactual predictions
    future_probs = np.zeros([len(game)+1, 4]) #array to story updated probabilities
    pred_err = np.zeros([len(game)+1, 4])  # hypothetical/counterfactual prediction errors
    pred_type = [''] * (len(game)+1)  # agent's prediction of which strategy is MOTIVATING the trustee's behavior
    pred_type[0] = np.random.choice(strat_type) #initialize randomly
    mb_pred_bx = np.zeros(len(game)+1) #model-based predictions
    p_strat[0, :] = .25  # Initialize the probability of using any strategy at chance for t=0
    future_probs[0, :] = .25

    for t in np.arange(len(game)):
        # The agent computes the behavior it would expect from someone using each of the 4 strategies.
        for i in np.arange(4):
            predicted_bx[t, i] = int(mp_model_ppsoe(inv=game.inv.iloc[t], mult=game.mult.iloc[t],
                                                    theta=folk_thry.loc[strat_type[i], 'theta'],
                                                    phi=folk_thry.loc[strat_type[i], 'phi']))
            #counterfactual predictions errors
            pred_err[t, i] = abs(game.ret.iloc[t] - predicted_bx[t, i])
            p_strat[t + 1, i] = p_strat[t, i] + alpha * ((1 - (pred_err[t, i] / (game.im.iloc[t]))) - p_strat[t, i])

        future_probs[t + 1, :] = utils.soft_max(p_strat[t + 1, :], tau)  # adjusted probabilities of p2 using each strategy
        pred_type[t + 1] = np.random.choice(strat_type, p=future_probs[t + 1, :])  # make probabilistic inference

        if pred_type[t] == 'gr':
            mb_pred_bx[t] = predicted_bx[t, 0]
        if pred_type[t] == 'ga':
            mb_pred_bx[t] = predicted_bx[t, 1]
        if pred_type[t] == 'ia':
            mb_pred_bx[t] = predicted_bx[t, 2]
        if pred_type[t] == 'mo':
            mb_pred_bx[t] = predicted_bx[t, 3]

    return mb_pred_bx[:57]

#%% Greed


def gr_model(inv, mult, theta, phi):
    inv = float(inv)
    mult = float(mult)
    theta = float(theta)
    phi = float(phi)
    return 0

#%% Guilt-Aversion


def ga_model(inv, mult, theta, phi):
    inv = float(inv)
    mult = float(mult)
    belMult = float(4) # what the investor believes the multiplier is
    exp = float(inv *2)
    theta = float(theta)
    phi = float(phi)

    total_amt = inv * mult
    choice_opts = np.arange(0, total_amt + 1)

    guilt = np.square(np.maximum((exp - choice_opts) / (inv * belMult), 0))

    own = total_amt - choice_opts

    utility = own - theta * guilt

    return choice_opts[np.where(utility == np.max(utility))[0][0]]
#%% Inequity-Aversion


def ia_model(inv, mult, theta, phi):
    inv = float(inv)
    mult = float(mult)
    theta = float(theta)
    phi = float(phi)

    total_amt = inv * mult
    choice_opts = np.arange(0, total_amt + 1)

    own = total_amt - choice_opts
    other = 10 - inv + choice_opts

    inequity = np.square(own / (own + other) - .5)

    utility = own - theta * inequity

    return choice_opts[np.where(utility == np.max(utility))[0][0]]
#%% Moral-Opportunism (van Baar et al. 2019)


# Moral Phenotyp model w Pre-Programmed Second Order Expectation (mp_model_ppsoe)
def mp_model_ppsoe(inv, mult, theta, phi):
    inv = float(inv)
    mult = float(mult)
    exp = float(2*inv)
    theta = float(theta)
    phi = float(phi)
    total_amt = inv*mult
    choice_opts = np.arange(0, total_amt+1)  # Only integers in strategy space (but not further discretized)
    own = total_amt-choice_opts
    other = 10 - inv + choice_opts
    own_share = own/total_amt  # Should be total_amt
    guilt = np.square(np.maximum((exp-choice_opts)/(inv*4), 0))
    inequity = np.square(own/(own+other) - .5)

    utility = theta*own_share - (1-theta)*np.minimum(guilt+phi, inequity-phi)

    return choice_opts[np.where(utility == np.max(utility))[0][0]]

