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

m1 = Lmer('ret_prcnt ~ pred_prcnt + (1|turker_id)', data=dat)
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
#%%
#%% Reformatting data
qDat = pd.read_csv('Data/questiondata_exp2.csv', header=None, index_col=None)
qDat.rename(columns={0:'turker', 1:'question', 2:'answer'}, inplace=True)
qDat = qDat.loc[qDat.turker.isin(dat.turker.unique())].reset_index(drop=True)
qq = qDat.pivot(index='turker', columns='question', values='answer')
qq.index = qq.index.set_names(['turker'])
qq = qq.reset_index()
qq.to_csv('Data/selfReport_data_exp2.csv',sep = ',', header=True)