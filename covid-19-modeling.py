import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.optimize as opt
import scipy.stats.distributions as dist
import seaborn as sns
import sklearn
import theano.tensor as tt
from matplotlib import cm
from matplotlib.colors import Normalize

from styling import *

# links to https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv
CASES_URL = "https://git.io/JvxHQ"
data = pd.read_csv(CASES_URL)

data = data.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='day', value_name='cases')
data['day'] = pd.to_datetime(data['day'])
data.set_index(['day', 'Country/Region'], inplace=True)
data.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)

# links to https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv
DEATHS_URL = "https://git.io/JvpP6"
deaths = pd.read_csv(DEATHS_URL)

deaths = deaths.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='day', value_name='deaths')
deaths['day'] = pd.to_datetime(deaths['day'])
deaths.set_index(['day', 'Country/Region'], inplace=True)
deaths.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)

total_cases = data.groupby("day").sum()
# total_cases = data.reset_index(1)
# total_cases = total_cases[total_cases['Country/Region'] == 'United Kingdom']
# or, for a list of countries:
# total_cases = total_cases[total_cases['Country/Region'].isin(['United Kingdom', 'US'])]
dt = total_cases.index
xx = np.arange(len(total_cases.index))
yy = total_cases['cases']

# deaths
total_deaths = deaths.groupby("day").sum()
dt_d = total_deaths.index
xx_d = np.arange(len(total_cases.index))
yy_d = total_deaths['deaths']

n_samples = 300
n_tune = 300
SEED = 1
N_COMPS = 2
N_DATA = len(xx)
xx2 = np.stack([xx, xx]).T
yy2 = np.stack([yy, yy]).T
with pm.Model() as model:
    k = pm.TruncatedNormal('k', mu=2 * yy[-1], sigma=yy[-1], lower=0, shape=N)
    sigma = pm.Exponential('sigma', lam=1 / 1e5, shape=N)
    dt = pm.Normal('dt', mu=30, sd=10, shape=N)
    tm = pm.Uniform('tm', lower=xx[0], upper=xx[-1], shape=N)
    yhat = k * pm.math.invlogit(np.log(81) / dt * (xx2 - tm))
    comps = pm.Normal.dist(mu=yhat, sigma=sigma, shape=(N, len(xx)))
    w = pm.Dirichlet('w', np.ones(N))
    obs = pm.Mixture('obs', w=w, comp_dists=comps, observed=yy)

    trace = pm.sample(draws=n_samples, tune=n_tune, random_seed=SEED, cores=3)

ALPHA = 0.05
params = np.vstack([trace['k'], trace['dt'], trace['tm']])

def plot_projection(ax, p=0.05, **kwargs):
    extended_xx = np.arange(len(xx) * 2)
    ddof = max(0, len(xx) - params.shape[0])
    thresh = dist.t.ppf(1 - p / 2, ddof)
    yvals = []
    param_mu = params.mean(axis=1)
    param_sd = params.std(axis=1)
    for t in (-thresh, 0, thresh):
        yvals.append(logistic(extended_xx, *(param_mu + t * param_sd)))
    ax.fill_between(extended_xx, yvals[0], yvals[2], **kwargs)
    ax.plot(extended_xx, yvals[1], '--', colors[0])

fig, ax = plt.subplots(figsize=(16, 8))
plot_projection(ax, ALPHA, cmap='plasma', alpha=0.2)

ax.plot(xx, yy, 'k')
fig.suptitle(f"Posterior Projection ({1-ALPHA:.1%} Confidence Interval)")
