import sys
sys.path.append('/Users/bryangonzalez/Documents/Dartmouth/Research/TheoryOfMind_Bx_experiment/Code/models/')
sys.path.append('/Users/bryangonzalez/Documents/Dartmouth/Research/TheoryOfMind_Bx_experiment/Code/')
import p3_cost_funcs
import comp_models
from utils import  folk_params, compute_aic, compute_bic, soft_max
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats.distributions import chi2
import pingouin as pg
from pymer4 import Lmer, Lm
import scipy
from scipy.optimize import least_squares

#%%
def likelihood_ratio(llmin, llmax):
        return 2 * (llmax - llmin)

dat = pd.read_csv('Data/bx_data.csv', index_col=0)
dat['cntr_trial'] = dat.trial - dat.trial.mean()
dat['trl_quadratic'] = dat.cntr_trial**2

#%% First stat. turker predictions regressed on Trustee bx

m1 = Lmer('ret ~ pred + (1|turker_id)', data=dat)
m1.fit()
m1anva = m1.anova()

#%%  comparing model w time vs. intercept
m3 = Lmer('turker_pe ~ 1 + (1 |turker_id)', data=dat)
m4 = Lmer('turker_pe ~ 1 + trial + (1 |turker_id)', data=dat)
m5 = Lmer('turker_pe ~ 1 + cntr_trial + trl_quadratic + (1 |turker_id)', data=dat)

m3.fit(REML=False)
m4.fit(REML=False)
ll_m3 = m3.logLike
ll_m4 = m4.logLike

LR = likelihood_ratio(ll_m3,ll_m4)
pea = chi2.sf(LR, 1)
#compare m4 to m5
m5.fit(REML=False)
ll_m5 = m5.logLike
LR = likelihood_ratio(ll_m4, ll_m5)
pea = chi2.sf(LR, 1)

#%% third stat: time regressed on errors x p2_strat

m2 = Lmer('turker_pe ~ cntr_trial + trl_quadratic + cntr_trial*p2_strat + trl_quadratic*p2_strat + (1|turker_id)', data=dat)
m2.fit(factors={'p2_strat': ['GR','GA', 'IA', 'MO']})
m2anva = m2.anova()
m2.post_hoc(marginal_vars='p2_strat')
m2anva_pg = pg.anova(data=dat, dv='turker_pe', between=['cntr_trial', 'p2_strat','trl_quadratic'])
m2.post_hoc(marginal_vars='trl_quadratic', grouping_vars='p2_strat')


#%%
#%% Reformatting data
qDat = pd.read_csv('Data/selfReport_data_exp2.csv', index_col=0)
'''
qDat['p2'] = np.nan
qDat['p3'] = np.nan
qDat['p2_strat'] = ''
qDat['mean_error'] = np.nan

for turk in qDat.turker.unique():
        qDat.loc[qDat.turker == turk, 'p3'] = dat.loc[dat.turker == turk, 'turker_id'].values[0]
        qDat.loc[qDat.turker == turk, 'p2'] = dat.loc[dat.turker == turk, 'trustee'].values[0]
        qDat.loc[qDat.turker == turk, 'p2_strat'] = dat.loc[dat.turker == turk, 'p2_strat'].values[0]
        qDat.loc[qDat.turker == turk, 'mean_error'] = dat.loc[dat.turker == turk, 'turker_pe'].mean()
 
qDat.rename(columns={
        "considerExpInputId":"considr_othrExp",
        'disappointInputId': "disappoint_othr",
        'fairInputId': "fairness",
        'friendsInputId': "be_friend",
        'greedInputId': "greedy",
        'knowInputId': "know_p2",
        'likeInputId': "liking",
        'mostOthersID': "like_mostOthrs",
       'selfDependInputId': "self_dependable",
        'selfDisappointInputId': "self_disappoint_othr",
        'selffairInputId': 'self_fairness',
       'selfgreedInputId': 'self_greedy',
        'selftrustInputId': 'self_trustwrthy',
        'similarInputId': 'similarity',
       'trustInputId': 'trust',
}, inplace = True)
qDat.to_csv('Data/selfReport_data_exp2.csv', sep=',')
'''

#%%
gr_qdat = qDat.loc[qDat.p2_strat == 'GR'].reset_index(drop=True)
ga_qdat = qDat.loc[qDat.p2_strat == 'GA'].reset_index(drop=True)
ia_qdat = qDat.loc[qDat.p2_strat == 'IA'].reset_index(drop=True)
mo_qdat = qDat.loc[qDat.p2_strat == 'MO'].reset_index(drop=True)

sns.set(style="darkgrid")

f, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")
ax = sns.kdeplot( ga_qdat.mean_error,ga_qdat.likeInputId, cmap="Greens",
                 shade=True, shade_lowest=False)
ax = sns.kdeplot( gr_qdat.mean_error, ia_qdat.likeInputId, cmap="Blues",
                 shade=True, shade_lowest=False)
# Add labels to the plot
green = sns.color_palette("Greens")[-2]
blue = sns.color_palette("Blues")[-2]
ax.text(2.5, 8.2, "observed guilt-averse", size=16, color=green)
ax.text(3.8, 4.5, "observed inequity-averse", size=16, color=blue)
plt.show()
#%%
f, ax = plt.subplots(figsize= (6,6))
sns.catplot(kind='bar', x= 'p2_strat', y='liking', data=qDat, ax=ax)
plt.savefig('/Users/bryangonzalez/Documents/Dartmouth/Research/TheoryOfMind_Bx_experiment/Figures/like_ratings.pdf')
f, ax = plt.subplots(figsize= (6,6))
sns.catplot(kind='bar', x= 'p2_strat', y='trust', data=qDat)
plt.savefig('/Users/bryangonzalez/Documents/Dartmouth/Research/TheoryOfMind_Bx_experiment/Figures/trust_ratings.pdf')
plt.show()

sns.catplot(kind='bar', x= 'p2_strat', y='be_friend', data=qDat)
plt.show()
plt.savefig('/Users/bryangonzalez/Documents/Dartmouth/Research/TheoryOfMind_Bx_experiment/Figures/be_frnds_ratings.pdf')
sns.catplot(kind='bar', x= 'p2_strat', y='fairness', data=qDat)
plt.show()


pos_imprssn = qDat.melt(id_vars=['p2','p3', 'mean_error', 'p2_strat'], value_vars=['fairness', 'be_friend','liking', 'trust'],
                        var_name='impression', value_name='rating')
f, ax = plt.subplots(figsize= (8,8))
sns.catplot(x='p2_strat', y='rating', hue='impression', kind='bar', data = pos_imprssn)
plt.savefig('/Users/bryangonzalez/Documents/Dartmouth/Research/TheoryOfMind_Bx_experiment/Figures/imprssn_ratings.pdf')
plt.show()


sns.jointplot(x='mean_error', y='liking', kind='reg', data=qDat.loc[qDat.p2_strat=='MO'])
pg.anova(data=qDat, dv='likeInputId', between='p2_strat')
pg.pairwise_tukey(data=qDat, dv='likeInputId', between='p2_strat')

dat['kept'] = dat['im'] - dat['ret']
dat['p2_keep_pct'] = (dat['inv']*dat['mult'] - dat['ret'])/(dat['inv'] * dat['mult'])
dat['Return_Pct'] = dat['ret']/(dat['inv']*dat['mult'])
dat['Expectation_Pct'] = dat['exp']/(dat['inv']*dat['mult'])
dat['Prediction_Pct'] = dat['pred']/(dat['inv']*dat['mult'])
dat['P1_Payoff'] = (10 - dat['inv']) + dat['ret']
dat['P2_Payoff'] = dat['inv']*dat['mult'] - dat['ret']
dat['P2_P1_Payoff_Ratio'] = dat['P2_Payoff'] - dat['P1_Payoff']
dat['P2_P1_Payoff_Ratio_Pct'] = (dat['P2_Payoff'] - dat['P1_Payoff'])/(dat['P2_Payoff'] + dat['P1_Payoff'])
dat['P1_P2_Payoff_Ratio_Pct'] = (dat['P1_Payoff'] - dat['P2_Payoff'])/(dat['P2_Payoff'] + dat['P1_Payoff']) #altruism
dat['P2_Return_Keep'] = (dat['ret'] - dat['kept'])/(dat['inv']*dat['mult'])
dat['P2_Sacrifice'] = (dat['exp'] - dat['P2_Payoff'])/(dat['inv']*dat['mult'])

qDat['P1_P2_Payoff_Ratio_Pct'] = np.nan

for turk in qDat.turker.unique():
        qDat.loc[qDat.turker == turk, 'P1_P2_Payoff_Ratio_Pct'] = dat.loc[dat.turker == turk, 'P1_P2_Payoff_Ratio_Pct'].mean()

sns.scatterplot(x='P1_P2_Payoff_Ratio_Pct', y='liking', data=qDat)
plt.show()

plt.subplots(figsize = (8,8))
sns.lmplot(x='P1_P2_Payoff_Ratio_Pct', y='liking', hue='p2_strat', hue_order=['IA', 'GR', 'GA', 'MO'], data=qDat)
plt.savefig('/Users/bryangonzalez/Documents/Dartmouth/Research/TheoryOfMind_Bx_experiment/Figures/altruism_liking.png')
plt.show()
f, ax = plt.subplots(figsize=(10,10))
sns.jointplot(x='P1_P2_Payoff_Ratio_Pct', y='liking', kind='reg', color='Orange', data=qDat.loc[qDat.p2_strat == 'GR'], ax=ax)
sns.jointplot(x='P1_P2_Payoff_Ratio_Pct', y='liking', kind='reg', color='Green', data=qDat.loc[qDat.p2_strat == 'GA'], ax=ax)
sns.jointplot(x='P1_P2_Payoff_Ratio_Pct', y='liking', kind='reg', color='Blue', data=qDat.loc[qDat.p2_strat == 'IA'], ax=ax)
sns.jointplot(x='P1_P2_Payoff_Ratio_Pct', y='liking', kind='reg', color='Red', data=qDat.loc[qDat.p2_strat == 'MO'], ax=ax)
plt.show()


plt.show()
qDat['p3'] = qDat.p3.astype(int)
qDat['p2_strat'] = qDat.p2_strat.astype(str)

m7 = Lmer('liking ~ P1_P2_Payoff_Ratio_Pct + p2_strat +  (1|p3)', data=qDat)
m7.fit(factors={'p2_strat':['GR', 'GA', 'IA', 'MO']})

f, ax = plt.subplots(figsize = (8,8))
#sns.lmplot(x='P1_P2_Payoff_Ratio_Pct', y='liking',data=qDat)
sns.jointplot(x='P1_P2_Payoff_Ratio_Pct', y='liking',kind='kde', data=qDat)
plt.savefig('/Users/bryangonzalez/Documents/Dartmouth/Research/TheoryOfMind_Bx_experiment/Figures/altruism_liking_allkde.png')
plt.show()
f, ax = plt.subplots(figsize = (8,8))
sns.catplot(x='p2_strat', y='P1_P2_Payoff_Ratio_Pct', kind='bar', data=qDat)
plt.savefig('/Users/bryangonzalez/Documents/Dartmouth/Research/TheoryOfMind_Bx_experiment/Figures/altruism_barall.png')
plt.show()

#%%
fits = pd.read_csv('Results/lstsqrs_fit2_p3_100iters.csv')
turk = 1
game = dat.loc[dat.turker_id == turk]
alpha = fits.loc[(fits.p3 == turk)&(fits.model == 'inverse motive'), 'param1'].values[0]
tau = fits.loc[(fits.p3 == turk)&(fits.model == 'inverse motive'), 'param2'].values[0]
def giveProbs(game, alpha, tau):
    strat_type = ['gr', 'ga', 'ia', 'mo', 'alt']  # The agent's heuristic 'state space'
    p_strat = np.zeros([len(game)+1, len(strat_type)])  # store the agent's belief of the prob that the trustee is using each strategy
    predicted_bx = np.zeros([len(game)+1, len(strat_type)]) #counterfactual predictions
    future_probs = np.zeros([len(game)+1, len(strat_type)]) #array to story updated probabilities
    pred_err = np.zeros([len(game)+1, len(strat_type)])  # hypothetical/counterfactual prediction errors
    pred_type = [''] * (len(game)+1)  # agent's prediction of which strategy is MOTIVATING the trustee's behavior
    pred_type[0] = np.random.choice(strat_type) #initialize randomly
    mb_pred_bx = np.zeros(len(game)+1) #model-based predictions
    p_strat[0, :] = 1/len(strat_type)  # Initialize the probability of using any strategy at chance for t=0
    future_probs[0, :] = 1/len(strat_type)
    folk_thry = folk_params

    for t in np.arange(len(game)):
        # The agent computes the behavior it would expect from someone using each of the 4 strategies.
        for i in np.arange(4):
            predicted_bx[t, i] = int(comp_models.mp_model_ppsoe(inv=game.inv.iloc[t], mult=game.mult.iloc[t],
                                                    theta=folk_thry.loc[strat_type[i], 'theta'],
                                                    phi=folk_thry.loc[strat_type[i], 'phi']))
            #counterfactual predictions errors
            pred_err[t, i] = abs(game.ret.iloc[t] - predicted_bx[t, i])
            p_strat[t + 1, i] = p_strat[t, i] + alpha * ((1 - (pred_err[t, i] / (game.im.iloc[t]))) - p_strat[t, i])
        predicted_bx[t, 4] = .1
        pred_err[t,4] = abs(game.P1_P2_Payoff_Ratio_Pct.iloc[t] - predicted_bx[t, 4])
        p_strat[t + 1, 4] = p_strat[t, 4] + alpha * ((1 - pred_err[t, 4]) - p_strat[t, 4])

        future_probs[t + 1, :] = soft_max(p_strat[t + 1, :], tau)  # adjusted probabilities of p2 using each strategy
        pred_type[t + 1] = np.random.choice(strat_type, p=future_probs[t + 1, :])  # make probabilistic inference

        if pred_type[t] == 'gr':
            mb_pred_bx[t] = predicted_bx[t, 0]
        if pred_type[t] == 'ga':
            mb_pred_bx[t] = predicted_bx[t, 1]
        if pred_type[t] == 'ia':
            mb_pred_bx[t] = predicted_bx[t, 2]
        if pred_type[t] == 'mo':
            mb_pred_bx[t] = predicted_bx[t, 3]
        if pred_type[t] == 'alt':
            mb_pred_bx[t] = game.im.iloc[t]

    probs = pd.DataFrame({
        'GR': future_probs[:57, 0],
        'GA': future_probs[:57, 1],
        'IA': future_probs[:57, 2],
        'MO': future_probs[:57, 3],
        'alt': future_probs[:57,4],
        'trial': np.arange(57),
        'p2': game.trustee,
        'p3': game.turker_id,
        'p2_strat': game.p2_strat
    })
    return probs


def giveProbs_imm(game, alpha, tau):
    strat_type = ['gr', 'ga', 'ia', 'mo']  # The agent's heuristic 'state space'
    p_strat = np.zeros([len(game) + 1, 4])  # store the agent's belief of the prob that the trustee is using each strategy
    predicted_bx = np.zeros([len(game) + 1, 4])  # counterfactual predictions
    future_probs = np.zeros([len(game) + 1, 4])  # array to story updated probabilities
    pred_err = np.zeros([len(game) + 1, 4])  # hypothetical/counterfactual prediction errors
    pred_type = [''] * (len(game) + 1)  # agent's prediction of which strategy is MOTIVATING the trustee's behavior
    pred_type[0] = np.random.choice(strat_type)  # initialize randomly
    mb_pred_bx = np.zeros(len(game) + 1)  # model-based predictions
    p_strat[0, :] = .25  # Initialize the probability of using any strategy at chance for t=0
    future_probs[0, :] = .25

    for t in np.arange(len(game)):
        # The agent computes the behavior it would expect from someone using each of the 4 strategies.
        for i in np.arange(4):
            predicted_bx[t, i] = int(comp_models.mp_model_ppsoe(inv=game.inv.iloc[t], mult=game.mult.iloc[t],
                                                    theta=folk_params.loc[strat_type[i], 'theta'],
                                                    phi=folk_params.loc[strat_type[i], 'phi']))
            # counterfactual predictions errors
            pred_err[t, i] = abs(game.ret.iloc[t] - predicted_bx[t, i])
            p_strat[t + 1, i] = p_strat[t, i] + alpha * ((1 - (pred_err[t, i] / (game.im.iloc[t]))) - p_strat[t, i])

        future_probs[t + 1, :] = soft_max(p_strat[t + 1, :], tau)  # adjusted probabilities of p2 using each strategy
        pred_type[t + 1] = np.random.choice(strat_type, p=future_probs[t + 1, :])  # make probabilistic inference

        if pred_type[t] == 'gr':
            mb_pred_bx[t] = predicted_bx[t, 0]
        if pred_type[t] == 'ga':
            mb_pred_bx[t] = predicted_bx[t, 1]
        if pred_type[t] == 'ia':
            mb_pred_bx[t] = predicted_bx[t, 2]
        if pred_type[t] == 'mo':
            mb_pred_bx[t] = predicted_bx[t, 3]
    probs = pd.DataFrame({
        'GR': future_probs[:57, 0],
        'GA': future_probs[:57, 1],
        'IA': future_probs[:57, 2],
        'MO': future_probs[:57, 3],
        'trial': np.arange(57),
        'p2': game.trustee,
        'p3': game.turker_id,
        'p2_strat': game.p2_strat
    })
    return probs
probs = pd.DataFrame(columns=['GR', 'GA', 'IA', 'MO', 'alt', 'trial','p2', 'p3','p2_strat'])

for turk in dat.turker_id.unique():
    game = dat.loc[dat.turker_id == turk]
    alpha = fits.loc[(fits.p3 == turk) & (fits.model == 'inverse motive'), 'param1'].values[0]
    tau = fits.loc[(fits.p3 == turk) & (fits.model == 'inverse motive'), 'param2'].values[0]
    p = giveProbs(game, alpha, tau)
    probs = probs.append(p)

probs_m = probs.melt(id_vars=['trial', 'p2_strat'], value_vars=['GR','GA','IA', 'MO', 'alt'], var_name='strat', value_name='prob')

sns.set_context("talk")
f, ax = plt.subplots(figsize=(10,10))
sns.relplot(x='trial',y='prob', hue='strat', col='p2_strat', col_wrap=2, kind='line', data=probs_m)
plt.savefig('/Users/bryangonzalez/Documents/Dartmouth/Research/TheoryOfMind_Bx_experiment/Figures/probAltruism.png')
plt.show()

#%%
imm_probs = pd.DataFrame(columns=['GR', 'GA', 'IA', 'MO', 'alt', 'trial','p2', 'p3','p2_strat'])

for turk in dat.turker_id.unique():
    game = dat.loc[dat.turker_id == turk]
    alpha = fits.loc[(fits.p3 == turk) & (fits.model == 'inverse motive'), 'param1'].values[0]
    tau = fits.loc[(fits.p3 == turk) & (fits.model == 'inverse motive'), 'param2'].values[0]
    p = giveProbs_imm(game, alpha, tau)
    imm_probs = imm_probs.append(p)

imm_probs_m = imm_probs.melt(id_vars=['trial', 'p2_strat'], value_vars=['GR','GA','IA', 'MO'], var_name='strat', value_name='prob')



sns.set_context("talk")
f, ax = plt.subplots(figsize=(10,10))
sns.relplot(x='trial',y='prob', hue='strat', col='p2_strat', col_wrap=2, kind='line', data=imm_probs_m)
#%%

#def imm_learn_space(game, n_states, alpha, tau):
turk = 1
game = dat.loc[dat.turker_id == turk]
fits = pd.read_csv('Results/lstsqrs_fit2_p3_100iters.csv', index_col=0)
alpha = fits.loc[(fits.p3 == turk) & (fits.model == 'inverse motive'), 'param1'].values[0]
tau = fits.loc[(fits.p3 == turk) & (fits.model == 'inverse motive'), 'param2'].values[0]


preds = comp_models.mentalState_space(game, alpha, tau, 10)
cpred = comp_models.imm_learn(game=game, alpha=alpha, tau=tau, folk_thry=folk_params)


n_states = 8
niters = 10

for turk in dat.turker_id.unique():
    game = dat.loc[dat.turker_id == turk].reset_index(drop=True)

    #Fit mental state space mode
    model = 'mental state space'
    print('Fitting '+model+' model to turker: '+str(turk)+'.')
    fitIters = np.zeros([niters, 5])
    for i in range(niters):
        param0 = [np.random.uniform(), np.random.uniform(low=.001)]
        fitIters[i,0:2] = [param0[0], param0[1]]
        result_lsq = least_squares(fun=p3_cost_funcs.mss_costfun, x0=param0,
                                   args=(game, n_states), bounds=([0,.001], [1,1]), diff_step=.05)

        [param1, param2] = result_lsq.x
        cost = result_lsq.cost
        fitIters[i, 2:5] = [param1, param2, cost]
    cost_selected = np.min(fitIters[:, 4])
    alpha = fitIters[fitIters[:,4]== cost_selected, 2][0]
    tau = fitIters[fitIters[:,4]== cost_selected, 3][0]
    sse = cost_selected* 2

#%%
'''
n_states = 10

decimals = int(np.log10(n_states) + 1)
phi = np.around(np.arange(-0.1, 0.10001, 0.2 / n_states).astype('float'), decimals=decimals)# [-.1,0,.1]
theta = np.around(np.arange(0, .50001, .5 / n_states).astype('float'), decimals=decimals)  # [0,.1,.2,.3,.4,.

p_strat = pd.DataFrame(data=1/(n_states**2), columns=theta, index=phi) # initialize uniform
counterfactuals = pd.DataFrame(columns=theta, index=phi)
pred_err = pd.DataFrame(columns=theta, index=phi)  # hypothetical/counterfactual prediction errors
pred_type =[]  # agent's prediction of which strategy is MOTIVATING the trustee's behavior
first_guess = (theta[np.random.randint(len(theta))], phi[np.random.randint(len(phi))])  # initialize randomly
pred_type.append(first_guess) # keep track of predictions
mb_pred_bx = np.zeros(len(game)+1)  # model predicted behavior
# string state-space into vector
tempStates = [[(thay, phii) for thay in theta] for phii in phi]
states = [x for l in tempStates for x in l] # [row1a, row1b, row1c..row2a, row2b...]


for t in np.arange(len(game)):
    if t == 0:
        mb_pred_bx[0] = int(comp_models.mp_model_ppsoe(inv=game.inv.iloc[0], mult=game.mult.iloc[0],
                                                       theta=first_guess[0], phi=first_guess[1]))
    else:
        mb_pred_bx[t] = int(comp_models.mp_model_ppsoe(inv=game.inv.iloc[0], mult=game.mult.iloc[0],
                                                       theta=future_state[0], phi=future_state[1]))

    for theyta in theta:
        for phee in phi:
            counterfactuals[theyta][phee] = comp_models.mp_model_ppsoe(inv=game.inv.iloc[t], mult=game.mult.iloc[t],
                                                                       theta=theyta, phi=phee)
            pred_err[theyta][phee] = abs(game.ret.iloc[t] - counterfactuals[theyta][phee])  # counterfactual predictions errors
            # probability update
            p_strat[theyta][phee] = p_strat[theyta][phee] + alpha * ((1 - (pred_err[theyta][phee] / (game.im.iloc[t]))) - p_strat[theyta][phee])
    #vectorize probabilities
    tempProb = [[p_strat[thay][phe] for thay in theta] for phe in phi]  # [[row1][row2]....
    prob = [x for l in  tempProb for x in l ] # [row1a, row1b, row1c..row2a, row2b...]

    temp = soft_max(prob, tau)
    future_state = states[np.random.choice(np.arange(len(states)), p=temp)]
    pred_type.append(future_state)
'''
#%%
sns.distplot(fits.loc[fits.model == 'inverse motive'].param1)
plt.show()
plt.hist(fits.loc[fits.model == 'inverse motive'].param1)
plt.show()