import numpy as np
import pandas as pd

import sys
sys.path.append('/Users/bryangonzalez/Documents/Dartmouth/Research/TheoryOfMind_Bx_experiment/Code/models/')
sys.path.append('/Users/bryangonzalez/Documents/Dartmouth/Research/TheoryOfMind_Bx_experiment/Code/')
import comp_models
import utils
import seaborn as sns

#GLOBALS
num_invs = 10
num_mults = 3
num_trials =  num_invs * num_mults
invs = np.arange(1, num_invs + 1)
mults = [2,4,6]#np.arange(1, num_invs + 1).tolist()
a = np.repeat(invs, num_mults).tolist()
b = mults * num_invs

#%% md

 #Simulation Trustee without noise using Moral Strategy Model (van Baar et al, 2019)

#%%

game = pd.DataFrame({
    'Investment': a,
    'Multiplier': b,
    'im': np.multiply(a, b),
    'GR_ret': [comp_models.mp_model_ppsoe(inv=a[x], mult=b[x], theta=.5, phi=0) for x in np.arange(len(a))],
    'GA_ret': [comp_models.mp_model_ppsoe(inv=a[x], mult=b[x], theta=0, phi=-.1) for x in np.arange(len(a))],
    'IA_ret': [comp_models.mp_model_ppsoe(inv=a[x], mult=b[x], theta=0, phi=.1) for x in np.arange(len(a))],
    'MO_ret': [comp_models.mp_model_ppsoe(inv=a[x], mult=b[x], theta=.13, phi=0) for x in np.arange(len(a))]
})
game_lng = game.melt(id_vars=['Investment', 'Multiplier', 'im'], value_vars=['GR_ret', 'GA_ret', 'IA_ret', 'MO_ret'],
                     var_name='Strategy', value_name='Return')
sns.catplot(x='Investment', y='Return', hue='Multiplier', col='Strategy', kind='point', data=game_lng)

#%% md

#Simulation Trustee with normally distributed noise using Moral Strategy Model (van Baar et al, 2019)
#N(canonical return , sd= | GA - IA | )

#%%

factor = .5
### ADDING NOISE N(exact strategy return, sd = |GA- IA|)
game_n = pd.DataFrame({
    'inv': a,
    'mult': b,
    'im': np.multiply(a, b),
    'canon_GR': game.GR_ret,
    'canon_GA': game.GA_ret,
    'canon_IA': game.IA_ret,
    'canon_MO': game.MO_ret,
    'GR_ret': np.random.exponential(1, num_trials),
    'GA_ret': (np.divide(abs((game['GA_ret'] - game['IA_ret'])), factor)) * np.random.randn() + game['GA_ret'],
    'IA_ret': (np.divide(abs((game['GA_ret'] - game['IA_ret'])), factor)) * np.random.randn() + game['IA_ret'],
    'MO_ret': (np.divide(abs((game['GA_ret'] - game['IA_ret'])), factor)) * np.random.randn() + game['MO_ret'],
    })

game_n_lng = game_n.melt(id_vars=['inv', 'mult', 'im'], value_vars=['GR_ret', 'GA_ret', 'IA_ret', 'MO_ret'],
                     var_name='Strategy', value_name='Return')
sns.catplot(x='inv', y='Return', hue='mult', col='Strategy', kind='point', data=game_n_lng)


#%%

game_n = game_n.sample(num_trials).reset_index(drop=True)

num_sim_games = 100

for i in np.arange(num_sim_games-1):
    c = pd.DataFrame({
        'inv': a,
        'mult': b,
        'im': np.multiply(a, b),
        'canon_GR': game.GR_ret,
        'canon_GA': game.GA_ret,
        'canon_IA': game.IA_ret,
        'canon_MO': game.MO_ret,
        'GR_ret': np.random.exponential(1, num_trials),
        'GA_ret': (np.divide(abs((game.GA_ret - game.IA_ret)), factor)) * np.random.randn() + game.GA_ret,
        'IA_ret': (np.divide(abs((game.GA_ret - game.IA_ret)), factor)) * np.random.randn() + game.IA_ret,
        'MO_ret': (np.divide(abs((game.GA_ret - game.IA_ret)), factor)) * np.random.randn() + game.MO_ret,

    })
    c = c.sample(num_trials)
    game_n = pd.concat([game_n,c], axis=0)


game_n_lng = game_n.melt(id_vars=['inv', 'mult', 'im'], value_vars=['GR_ret', 'GA_ret', 'IA_ret', 'MO_ret'],
                     var_name='Strategy', value_name='Return')
sns.catplot(x='inv', y='Return', hue='mult', col='Strategy', kind='point', data=game_n_lng)


