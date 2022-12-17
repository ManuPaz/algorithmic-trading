
import pickle
import os

import pandas as pd

os.chdir("../../")
from src.functions import metrics
from src.plots import portfolio_plots
from src.utils import work_with_nas,save_results
import numpy as np
annualized=252
DAILY_RISK_FREE_RETURN= 0.02 / annualized
if __name__=="__main__":
    with open("resources/stocks_summary.obj","rb") as stocks_summary_file:
        stocks_summary=pickle.load(stocks_summary_file)
    with open("resources/other_equities.obj","rb") as other_equities_file:
        other_equities=pickle.load(other_equities_file)
    other_equities.index=work_with_nas.drop_rows_with_nas(other_equities.index, columns="all")
    index= other_equities.index
    returns = index.pct_change().dropna()
    for i,column in enumerate(index.columns):
        kwargs = {"from_": index.dropna(subset=[column]).index[0], "to": index.dropna(subset=[column]).index[-1], "annualized": annualized,"DAILY_RISK_FREE_RETURN": DAILY_RISK_FREE_RETURN, "title": column}
        save_results.all_backtesting_results(returns[column], column, "",**kwargs)