# Import numpy library
import numpy as np

# Define function to calculate Sharpe ratio
def sharpe_ratio(original_prices, risk_free_return):
    returns = (original_prices[1:] / original_prices[:-1]) - 1
    expected_return = np.mean(returns)
    standard_deviation = np.std(returns)
    return (expected_return - risk_free_return) / standard_deviation

# Define function to calculate Sortino ratio
def sortino_ratio(original_prices, risk_free_return):
    returns = (original_prices[1:] / original_prices[:-1]) - 1
    expected_return = np.mean(returns)
    excess_returns = returns - risk_free_return
    negative_deviation = np.std(excess_returns[excess_returns < 0])
    return (expected_return - risk_free_return) / negative_deviation

# Define function to calculate analyzed return
def analyzed_return(original_prices):
    n=len(original_prices)/252
    return ((original_prices[-1] / original_prices[0])**(1/n) - 1) * 100

# Define function to calculate maximum drawdown
def maximum_drawdown(original_prices):
    account_balance = original_prices[1:] / original_prices[:-1]
    maximum_drawdown = (np.max(account_balance) - account_balance) / np.max(account_balance)
    return np.max(maximum_drawdown) * 100
