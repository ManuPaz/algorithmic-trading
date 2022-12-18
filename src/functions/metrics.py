# Import numpy library
import numpy as np
import math
import pandas as pd

def all_metrics_array(returns, risk_free_return, annualized=1):
    m=all_metrics(returns, risk_free_return, annualized)
    return [m["sharpe_ratio"],m["sortino_ratio"],m[ "annualized_return %"],m["maximum_drawdown"]]
def all_metrics(returns, risk_free_return, annualized=1):
    prices=(1 + returns).cumprod()
    return {"sharpe_ratio":sharpe_ratio(returns, risk_free_return, annualized),
            "sortino_ratio":sortino_ratio(returns, risk_free_return, annualized),
            "annualized_return %":annualized_return(returns,annualized),
            "maximum_drawdown":maximum_drawdown(prices)

            }
# Define function to calculate Sharpe ratio
def sharpe_ratio(returns, risk_free_return, annualized=1):
    expected_return = np.mean(returns)
    standard_deviation = np.std(returns)
    return math.sqrt(annualized)*(expected_return - risk_free_return) / standard_deviation
# Define function to calculate Sortino ratio
def sortino_ratio(returns, risk_free_return, annualized=1):
    expected_return = np.mean(returns)
    excess_returns = returns - risk_free_return
    negative_deviation = np.std(excess_returns[excess_returns < 0])
    return math.sqrt(annualized)*(expected_return - risk_free_return) / negative_deviation

# Define function to calculate analyzed return
def annualized_return(returns,annualized=1):
    n=len(returns)/annualized
    return (np.cumprod(1+returns)[-1]**(1/n)-1)

# Define function to calculate maximum drawdown
def maximum_drawdown(original_prices):
    maximos=pd.Series(original_prices).cummax()
    return -min((original_prices-maximos)/maximos)
