import pandas as pd
from cvxopt import matrix, solvers
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
import pandas as pd
import numpy as np
import pypfopt.black_litterman as black_litterman
from pyhrp.hrp import dist, linkage, tree, _hrp
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import logging.config
logging.config.fileConfig('resources/logging.conf')
logger = logging.getLogger('api_error')
def same_allocation(returns):
    symbols_per_day = returns.shape[1] - returns.isna().sum(axis=1)
    returns_portfolio = returns.multiply(1 / symbols_per_day, axis="rows").replace(np.nan, 0)
    returns_portfolio = returns_portfolio.sum(1)
    return  returns_portfolio


def portfolio_returns(weights,returns):
    r=returns.loc[:,weights.index]
    returns_portolio = np.dot(r.replace(np.nan, 0), weights)
    serie = pd.Series(returns_portolio)
    serie.index = returns.index
    return serie
def hierarchical_allocation(returns,linkage_t="single",risk_free_rate=0):
    r=returns.dropna(axis=1)
    cov, cor = r.cov(), r.corr()
    links = linkage(dist(cor.values), method=linkage_t)
    node = tree(links)

    rootcluster = _hrp(node, cov)
    plt.figure(figsize=(10,20))
    ax=dendrogram(links, orientation="left",labels=list(r.columns))
    plt.show()
    weights= pd.Series(rootcluster.assets)


    dic_weights =weights.to_dict()
    print("Weights herarhical clustering: {},{}".format(dic_weights,weights.sum()))
    weights=weights/weights.sum()
    returns_portolio = np.dot(r.replace(np.nan, 0), weights.values.reshape(-1))
    serie = pd.Series(returns_portolio)
    serie.index = returns.index
    return serie,pd.Series(dic_weights)


def black_litterman_allocation(stocks_summary, returns, market_returns, absolute_views=None, relative_views=None, risk_free_rate=0.02 / 252, **kwargs):
    symbols=list(returns.columns)
    if kwargs["use_market_caps"]:
        m_caps,symbols = stocks_summary.get_market_caps(symbols, returns.index[0], stocks_summary)
    r=returns.loc[:, symbols]
    if absolute_views is not None:
        absolute_views={e:absolute_views[e] for e in absolute_views.keys() if e in symbols}
    if relative_views is not None:
        relative_views=[e for e in relative_views if set(e.keys()).issubset(set(symbols)) ]
    filas1=  0 if  absolute_views is  None else len(absolute_views.keys())
    filas2= 0  if  relative_views is  None else len(relative_views)
    P=np.zeros((filas1+filas2,len(symbols)))
    Q=np.zeros((filas1+filas2))
    desp=0
    if absolute_views is not None:
        for i,view in enumerate(absolute_views.keys()):
            Q[i]=absolute_views[view]
            P[i,symbols.index(view)]=1 if absolute_views[view]!=0 else 0
        desp=i+1
    if relative_views is not None:
        for i,elem in enumerate(relative_views):
            Q[i+desp]=abs(elem[list(elem.keys())[0]])
            suma=len( elem.values())/2
            for symbol in elem.keys():
                P[desp+i,symbols.index(symbol)]=1/suma if elem[symbol]>0 else -1/suma

    cov=r.cov()

    delta= market_implied_risk_aversion(market_returns, risk_free_rate=risk_free_rate)
    print("Delta {}".format(delta))
    if delta<0.2:
        delta=0.2
    if kwargs["use_market_caps"]:
        bl = BlackLittermanModel(cov,Q=Q,P=P, pi="market", risk_aversion=delta, market_caps=m_caps.to_dict())
    else:
        bl = BlackLittermanModel(cov,Q=Q,P=P, pi="equal", risk_aversion=delta, )
    bl.bl_weights()
    weights = bl.clean_weights()
    weights=np.array( list(weights.values()))
    weights=weights/abs(weights).sum()
    returns_portolio = np.dot(r.replace(np.nan, 0),weights)
    serie = pd.Series(returns_portolio)
    serie.index = returns.index
    dic_weights={symbols[i]:weights[i] for i in range(len(weights))}
    print("Weights Black-Litermann {}".format(dic_weights))
    return  serie,pd.Series(dic_weights)
def markowitz(returns):

    # Calcular varianzas y covarianzas
    varianzas = returns.var()
    covarianzas = returns.cov()

    # Crear matrices de entrada para el problema de optimizaci??n
    n = len(varianzas)
    S = matrix(covarianzas.values)
    pbar = matrix(returns.mean())
    G = matrix(0.0, (n, n))
    G[::n + 1] = -1.0
    h = matrix(0.0, (n, 1))
    A = matrix(1.0, (1, n))
    b = matrix(1.0)
    solvers.options['show_progress'] = False
    # Resolver problema de optimizaci??n
    sol = solvers.qp(S, -pbar, G, h, A, b)
    weights=np.array(sol['x']).reshape(-1)
    dic_weights = {returns.columns[i]: weights[i] for i in range(len(weights))}
    print("Weights Markowitz: {}".format(dic_weights))
    # Imprimir asignaci??n de activos en la cartera
    returns_portolio=np.dot(returns.replace(np.nan, 0), weights)
    serie = pd.Series(returns_portolio)
    serie.index= returns.index

    return  serie,pd.Series(dic_weights)
def market_implied_risk_aversion(market_returns,  risk_free_rate=0.02):

    if not isinstance(market_returns , (pd.Series, pd.DataFrame)):
        raise TypeError("Please format market_prices as a pd.Series")

    r = market_returns.mean()
    var = market_returns.var()
    return (r - risk_free_rate) / var


