import os
os.chdir("../../")
from src.utils import save_results
import numpy as np
import pickle

annualized=252
DAILY_RISK_FREE_RETURN = 0.02 / annualized
MAX_RETURN_DAILY = 0.4
import src.utils.load_config as load_config
general_config = load_config.general_config()
import src.functions.financial_functions as financial_functions
import src.utils.time_utils as time_utils
SAVE_RESULTS=True

if __name__ == "__main__":
    with open("resources/stocks_summary.obj", "rb") as stocks_summary_file:
        stocks_summary = pickle.load(stocks_summary_file)

    with open("resources/other_equities.obj", "rb") as other_equities_file:
        other_equities = pickle.load(other_equities_file)


    prices = stocks_summary.get_all_prices(list(stocks_summary.stock_objects.keys()))
    returns = prices.loc[:, (slice(None), "adj_close")].droplevel(1, axis=1).pct_change().iloc[1:,:]
    to_high_returns = stocks_summary.clean_data(returns, max_return=MAX_RETURN_DAILY)

    returns_portfolio=financial_functions.same_allocation(returns)
    kwargs={"from_":returns.index[0],"to":returns.index[-1],"annualized":annualized,"DAILY_RISK_FREE_RETURN":DAILY_RISK_FREE_RETURN,"title":"same allocation"}
    if SAVE_RESULTS:
        save_results.all_backtesting_results(returns_portfolio,"dummy","Dummy algorithm with the same allocation for each stock",**kwargs)

    returns_portfolio,_=financial_functions .markowitz(returns)
    returns_portfolio
    kwargs["title"]="markowitz"
    if SAVE_RESULTS:
        save_results.all_backtesting_results(returns_portfolio, "markowitz","Markowtiz allocation with with weights summing 1", **kwargs)

    returns_portfolio,_=financial_functions .black_litterman_allocation(stocks_summary, returns, other_equities.index["sp500"].pct_change().dropna(),
                                                                        absolute_views=general_config["black_literman_absolute_views"],
                                                                        relative_views=general_config["black_literman_relative_views"],
                                                                        risk_free_rate=DAILY_RISK_FREE_RETURN,
                                                                        use_market_caps=True)
    kwargs["title"] = "black_litterman"
    if SAVE_RESULTS:
        save_results.all_backtesting_results(returns_portfolio, "black_litterman",
                                         "Black Litterman allocation with with weights summing 1", **kwargs)
    kwargs["title"] = "hierarchical risk parity"
    returns_portfolio,_=financial_functions.hierarchical_allocation(returns)
    if SAVE_RESULTS:
        save_results.all_backtesting_results(returns_portfolio, "hierarchical allocation",
                                             "Hierarchical Risk Parity allocation with with weights summing 1", **kwargs)