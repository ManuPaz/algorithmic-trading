import pymc3 as pm
import  matplotlib.pyplot as plt
import theano as th
import os
import pandas as pd
import seaborn as sns
import logging
logging.basicConfig(level=logging.INFO)
theano_logger=logging.getLogger('theano')
theano_logger.setLevel(logging.INFO)
def bayesian_cointegration_regresion(data1,data2,indice):
    with pm.Model() as model:
        # inject external stock data
        stock1 = th.shared(data1)
        stock2 = th.shared(data2)

        # define our cointegration variables
        beta_sigma = pm.Exponential('beta_sigma', 50.)
        beta = pm.GaussianRandomWalk('beta', sd=beta_sigma,
                                     shape=data1.shape[0])

        # with our assumptions, cointegration can be reframed as a regression problem
        stock2_regression = beta * stock1
        # Assume prices are Normally distributed, the mean comes from the regression.
        sd = pm.HalfNormal('sd', sd=.1)
        likelihood = pm.Normal('y',
                               mu=stock2_regression,
                               sd=sd,
                               observed=stock2)
    with model:
        stock1.set_value(data1)
        stock2.set_value(data2)
        trace = pm.sample(1000, tune=400, cores=1,progressbar=True)
    rolling_beta = trace[beta].T.mean(axis=1)
    plt.figure(figsize=(18, 8))
    ax = plt.gca()
    plt.title("Beta Distribution over Time")
    pd.Series(rolling_beta, index=indice).plot(ax=ax, color='r', zorder=1e6, linewidth=2)
    for orbit in trace[beta][:500]:
        pd.Series(orbit, index=indice).plot(ax=ax, color=sns.color_palette()[0], alpha=0.05)
    plt.legend(['Beta Mean', 'Beta Orbit'])
    # plt.savefig("beta distrib.png")
    plt.show()
    return rolling_beta