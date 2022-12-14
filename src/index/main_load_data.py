import os
import pandas as pd

import time
import pickle

print(os.getcwd())
if os.getcwd().split("\\")[-1] == "index":
    print("Changing base directory ...")
    os.chdir("../..")

from src.functions import apis, stocks_module
import src.utils.load_config as load_config

secret_config = load_config.secret_config()
general_config = load_config.general_config()
import logging.config
only_prices=False
logging.config.fileConfig('resources/logging.conf')
logger = logging.getLogger('general')
pd.set_option('display.max_columns', 500)
if __name__ == '__main__':

    tiempo1 = time.time()
    stocks_summary = None
    file_stocks="resources/algorithm5/stocks_summary.obj"
    file_index="resources/algorithm5/other_equities.obj"
    if os.path.isfile(file_stocks):
        with open(file_stocks, "rb") as file:
            stocks_summary = pickle.load(file)
    alpha_vantage = apis.AlphaVantage(secret_config["alpha_vantage_key"])
    finhub = apis.Finhub(secret_config["finhub_key"])
    polygon = apis.Polygon(secret_config["polygon_key"])
    yFinance = apis.YFinance()
    pandas_data_reader = apis.PandasDataReader()
    indice_name=general_config["index_name"]
    if indice_name=="russell2000":
        tickers = list(pd.read_csv("data/russell2000_stockslisted.csv")["Symbol"].values)
    else:
        try:
            tickers=yFinance.get_tickers(indice_name)
            tickers.sort()
            logger.info("Numero de tickers {}".format(len(tickers)))
        except Exception as e:
            logger.error(e)

    index = pandas_data_reader.get_index(["sp500", "nasdaq100",],from_=general_config["FROM"],to=general_config["TO"])
    economic_calendar = alpha_vantage.get_macroeconomic_data()
    other_equities=stocks_module.OtherEquities(index=index,economic_calendar=economic_calendar)
    with open(file_index, "wb") as file_equities:
        pickle.dump(other_equities, file_equities, protocol=pickle.HIGHEST_PROTOCOL)

    earnings_calendar = alpha_vantage.get_earnings_calendar(horizon="3month")
    if stocks_summary is not None:

        stocks_summary.earnings_calendar=earnings_calendar
    if general_config["GET_GLOBAL_SUMMARY"]:

        general_summary = None
        if len(tickers) < 40:
            general_summary = yFinance.get_actual_summary(tickers,indice_name)
        stocks_profile=yFinance.get_stock_profile(tickers,indice_name)
        financials_summary = yFinance.get_actual_financial_data(tickers,indice_name)
        financials_summary_finhub = finhub.get_basic_financials_current(tickers,indice_name)

        logger.info("First query time: {}".format(time.time() - tiempo1))
        if stocks_summary is None:
            stocks_summary = stocks_module.StocksSummary( general_summary=general_summary,
                                                         fundamental_summary=financials_summary,
                                                         earnings_calendar=earnings_calendar,
                                                         fundamental_summary_finhub=financials_summary_finhub,
                                                         stock_profile=stocks_profile)
        else:
            stocks_summary.update( general_summary=general_summary,
                                  fundamental_summary=financials_summary,
                                  earnings_calendar=earnings_calendar,
                                  fundamental_summary_finhub=financials_summary_finhub,
                                  stock_profile=stocks_profile)

    elif  stocks_summary is None:
        stocks_summary = stocks_module.StocksSummary()

    with open(file_stocks, "wb") as file:
        pickle.dump(stocks_summary, file, protocol=pickle.HIGHEST_PROTOCOL)
    tiempo1 = time.time()
    tickers=general_config["tickers"]
    #tickers=stocks_summary.fundamental_summary_complete.index.get_level_values(0)
    for ticker in tickers:
        logger.info("Getting {} info ...".format(ticker))
        prices = yFinance.get_stock_prices(ticker, interval=general_config["PRICES_INTERVAL"],
                                           range=general_config["PRICES_RANGE"], from_=general_config["FROM"],
                                           to=general_config["TO"])
        if not only_prices:
            indicators=finhub.get_technical_indicators(ticker,indicators=["ema","sma","macd","adx","rsi"],resolution=general_config["PRICES_INTERVAL"],from_=general_config["FROM"],to=general_config["TO"])
            earnings_a, earnings_q, earnings_trend = yFinance.get_earnings(ticker)
            twitter, reddit = finhub.get_sentiment(ticker)
            sentiment=alpha_vantage.get_sentiments(ticker)
            annual_results = finhub.get_basic_financials(ticker)
            quarterly_resuts = finhub.get_basic_financials(ticker, freq="quarterly")

            recomendation = finhub.get_recomendation_trends(ticker)
            calls, puts = yFinance.get_options(ticker)
            kwargs={"sentiment":pd.DataFrame(),"twitter":twitter,"reddit":reddit,
                    "annual_eps":earnings_a,"quarterly_eps":earnings_q,"trending_eps":earnings_trend,
                    "puts":puts,"calls":calls,"quarterly_financials":quarterly_resuts,
                    "annual_financials":annual_results,"recomendation":recomendation,"indicators":indicators}

            stock = stocks_module.Stock(name=ticker, prices=prices,**kwargs )
            stocks_summary.stock_objects[ticker] = stock
        elif ticker not in stocks_summary.stock_objects.keys():
            print(ticker)
            stock = stocks_module.Stock(name=ticker, prices=prices)
            stocks_summary.stock_objects[ticker] = stock

    with open(file_stocks, "wb") as file:
            pickle.dump(stocks_summary, file, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Second query time: {}".format(time.time() - tiempo1))
