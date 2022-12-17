import datetime
import os
import json
import pandas as pd
from src.functions import metrics

os.chdir("../../")
from src.utils import save_results
import numpy as np
import logging.config

logging.config.fileConfig('resources/logging.conf')
logger = logging.getLogger('general')
import pickle
from src.functions.financial_functions import markowitz, same_allocation, black_litterman_allocation, \
    hierarchical_allocation, portfolio_returns

annualized = 252
DAILY_RISK_FREE_RETURN = 0.02 / annualized
MAX_RETURN_DAILY = 0.4
import src.utils.load_config as load_config

general_config = load_config.general_config()
import src.functions.financial_functions as financial_functions
import src.utils.time_utils as time_utils
from src.utils.clean_df import drop_columns_with_many_nas, drop_zeros
from src.utils.save_results import save_train_test
SAVE_RESULTS = True





if __name__ == "__main__":
    with open("resources/stocks_summary.obj", "rb") as stocks_summary_file:
        stocks_summary = pickle.load(stocks_summary_file)

    with open("resources/other_equities.obj", "rb") as other_equities_file:
        other_equities = pickle.load(other_equities_file)

    prices = stocks_summary.get_all_prices(list(stocks_summary.stock_objects.keys()))
    returns = prices.loc[:, (slice(None), "adj_close")].droplevel(1, axis=1).pct_change().iloc[1:, :]
    intervals = time_utils.generate_random_intervals(returns.index[0], returns.index[-1], train_duration=120,
                                                     test_duration=60,
                                                     frequency="d")

    baseselines_index = other_equities.index
    returns_index = baseselines_index.pct_change()
    returns_index = drop_columns_with_many_nas(returns_index)
    algorithms = ["same_allocation", "markowtiz", "black_literman", "hierarchical_risk_parity"] + list(
        returns_index.columns)
    metrics = ["sharpe_ratio", "sortino_ratio", "return", "maximum_drawdown", ]
    index = pd.MultiIndex.from_frame(intervals.loc[:, ["start_train", "end_train"]], names=["start", "end"])
    df_results_train = pd.DataFrame(index=index, columns=pd.MultiIndex.from_product([algorithms, metrics]))
    index = pd.MultiIndex.from_frame(intervals.loc[:, ["start_test", "end_test"]], names=["start", "end"])
    df_results_test = pd.DataFrame(index=index, columns=pd.MultiIndex.from_product([algorithms, metrics]))
    kwargs = {"DAILY_RISK_FREE_RETURN": DAILY_RISK_FREE_RETURN, "annualized": annualized}
    dic_weights = {"markowitz": {}, "black_literman": {}, "hierarchical_risk_parity": {}}
    for interval in intervals.iterrows():
        interval = interval[1]
        returns_train = returns.loc[interval["start_train"]:interval["end_train"]]
        returns_test = returns.loc[interval["start_test"]:interval["end_test"]]
        returns_train = drop_columns_with_many_nas(returns_train)
        returns_train = drop_zeros(returns_train)
        returns_test = drop_columns_with_many_nas(returns_test)

        logger.info(" Interval train {}-{}, Shape train {}, shape test {}".format(interval["start_train"],
                                                                                  interval["end_train"],
                                                                                  returns_train.shape,
                                                                                  returns_test.shape))
        columns = list(set(returns_test).intersection(set(returns_train)))
        returns_train = returns_train.loc[:, columns]
        returns_test = returns_test.loc[:, columns]

        portfolio_train_markowitz, w_markowitz = markowitz(returns_train)
        dic_weights["markowitz"][str((interval["start_test"], interval["end_test"]))] = w_markowitz.to_dict()
        save_train_test(interval, portfolio_train_markowitz, returns_test, w_markowitz, "markowtiz", df_results_train,
                        df_results_test,
                        **kwargs)
        market_ret = returns_index.loc[interval["start_train"]:interval["end_train"], "sp500"]
        portfolio_train_black_litterman, w_black_literman = black_litterman_allocation(stocks_summary, returns_train,
                                                                                       market_ret,
                                                                                       absolute_views=general_config[
                                                                                           "black_literman_absolute_views"],
                                                                                       relative_views=general_config[
                                                                                           "black_literman_relative_views"],
                                                                                       risk_free_rate=DAILY_RISK_FREE_RETURN,
                                                                                       use_market_caps=True)
        dic_weights["black_literman"][str((interval["start_test"], interval["end_test"]))] = w_black_literman.to_dict()
        save_train_test(interval, portfolio_train_black_litterman, returns_test, w_black_literman, "black_literman",
                        df_results_train,
                        df_results_test, **kwargs)

        portfolio_train_hrp, w_hrp = hierarchical_allocation(returns_train)
        dic_weights["hierarchical_risk_parity"][str((interval["start_test"], interval["end_test"]))] = w_hrp.to_dict()
        save_train_test(interval, portfolio_train_hrp, returns_test, w_hrp, "hierarchical_risk_parity",
                        df_results_train, df_results_test, **kwargs)

        portfolio_train_same_allocation = same_allocation(returns_train)
        portfolio_test_same_allocation = same_allocation(returns_test)
        save_results.multiple_intervals_simulation_save_results(df_results_train, portfolio_train_same_allocation,
                                                                "same_allocation",
                                                                interval["start_train"], interval["end_train"],
                                                                **kwargs)
        save_results.multiple_intervals_simulation_save_results(df_results_test, portfolio_test_same_allocation,
                                                                "same_allocation",
                                                                interval["start_test"], interval["end_test"],
                                                                **kwargs)

        for col in returns_index.columns:
            r = returns_index.loc[interval["start_train"]:interval["end_train"], col]
            save_results.multiple_intervals_simulation_save_results(df_results_train, r, col, interval["start_train"],
                                                                    interval["end_train"], **kwargs)
            r = returns_index.loc[interval["start_test"]:interval["end_test"], col]
            save_results.multiple_intervals_simulation_save_results(df_results_test, r, col, interval["start_test"],
                                                                    interval["end_test"], **kwargs)

    dir = "reports/backtesting/simulations/algorithm2"
    if not os.path.isdir(dir):
        os.makedirs(dir)
    with open(os.path.join(dir, "weights.json"), "w") as file:
        json.dump(dic_weights, file)
    df_results_train.loc[("mean", "mean"), :] = df_results_train.mean().values
    df_results_test.loc[("mean", "mean"), :] = df_results_test.mean().values
    df_results_test.loc[("mean", "mean"), (slice(None), "return")] = df_results_test.loc[("mean", "mean"), (
    slice(None), "return")].values * 252
    df_results_train.loc[("mean", "mean"), (slice(None), "return")] = df_results_train.loc[("mean", "mean"), (
    slice(None), "return")].values * 252
    df_results_train.to_excel(os.path.join(dir, str(datetime.date.today()) + "_train.xlsx"), index=True)
    df_results_test.to_excel(os.path.join(dir, str(datetime.date.today()) + "_test.xlsx"), index=True)
