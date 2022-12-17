
import os
import numpy as np
import pandas as pd
import src.utils.load_config as load_config
from src.functions import metrics
general_config = load_config.general_config()
from scipy import stats as st
from src.plots import  portfolio_plots
from src.functions.financial_functions import portfolio_returns

def save_train_test(interval, returns_portfolio_train, returns_test, weights_train, name, df_results_train,
                    df_results_test, **kwargs):
    returs_portfolio_test = portfolio_returns(weights_train, returns_test)

    multiple_intervals_simulation_save_results(df_results_train, returns_portfolio_train, name,
                                                            interval["start_train"], interval["end_train"],
                                                            **kwargs)
    multiple_intervals_simulation_save_results(df_results_test, returs_portfolio_test, name,
                                                            interval["start_test"], interval["end_test"],
                                                            **kwargs)
def all_backtesting_results(returns_portfolio,name,description,**kwargs):
    portfolio_plots.plot_returns((1 + returns_portfolio).cumprod(), title=kwargs["title"], )
    metrics_array=metrics.all_metrics_array(returns_portfolio,
                        risk_free_return=kwargs["DAILY_RISK_FREE_RETURN"],
                        annualized=kwargs["annualized"])
    if not os.path.isdir(general_config["all_backtesting_results_path"]):
        os.makedirs(general_config["all_backtesting_results_path"])
    file=os.path.join(general_config["all_backtesting_results_path"],"all_results.xlsx")
    if os.path.isfile(file):
        data=pd.read_excel(file,index_col=0)
    else:
        data=pd.DataFrame(columns=["description","free_rate","from","to","period","sharpe_ratio","sortino_ratio","annualized_return","maximum_drawdown"])
    period=float(st.mode(np.diff(returns_portfolio.index))[0]/1000000000/3600)
    if period>=24:
        period=str(period/24)+" dias"
    else:
        period=str(period)+" horas"
    data.loc[name,:]=[description,kwargs["DAILY_RISK_FREE_RETURN"]*kwargs["annualized"],kwargs["from_"],kwargs["to"],period]+metrics_array
    data.to_excel(file,index=True)


def multiple_intervals_simulation_save_results(df_results,returns_portfolio,name,interval_start,interval_end,**kwargs):
    metrics_array = metrics.all_metrics_array(returns_portfolio,
                                              risk_free_return=kwargs["DAILY_RISK_FREE_RETURN"],
                                              annualized=kwargs["annualized"])
    df_results.loc[(interval_start,interval_end),(name,["sharpe_ratio","sortino_ratio","return","maximum_drawdown"])]=metrics_array