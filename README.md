# Algorithmic trading
In this project there are several scripts to download data from free financials API's such as Finhub or Yahoo Finance and use them to backtest diferent algorithmic strategies, or to apply filters with a programatic market screener.
## Proyect structure
With  **index/main_load_data** data is downloaded and saved in  **StocksSummary** and then in a pickle object. When   **index/main_load_data** is executed with different stocks the pickle object is updated so previuos data is not deleted, just modified or extended with new stocks.
**StocksSummary** keeps differente info, but mainly: on one side actual stocks info to use in the screener and on the other side a list of **Stock** objects, where each stock keeps its historic financial series.
* **index/main_load_data**. To download data from the API's.  To set the tickers you want to download you can set the next properties in  **general_properties.yaml** file:
  * **tickers**. Tickers to get the financial series: historic prices, historic eps, historic financials, options, sentiment, recomendations ...
  * **index_name**. The index to get current data. A dataframe with one row per stock and several columns with ratios and info is created. If the script is executed several times the stocks of the nwe index are added to the dataframe, but the previous are not deleted.
* **main_screener**. Screener to apply several filters on financial and stocks data on the dataframe created with the **index/main_load_data**.
* **algorithms**. Different algorithms using the stock info on the stock series.
* **resources/secret_properties.yaml**. Here are the API's keys. 
  * **alpha_vantage_key**
  * **polygon_key**
  * **finhub_key**
## Algorithms
### Porfolio allocation
+ **Algorithm1** simply takes a period of time and generates an excel file with the results of different portfolios: same allocation, markowitz, hierarchical risk parity, black litterman, and several reference indices. For the black litterman portfolio, you need to specify in **general_config.yaml** the relative and absolute views of the symbols you will use .
* **Algorithm2** is the same as **algorithm1** but pairs of training and testing intervals (testing is always next to training) are generated with some randomness using a training  interval length and testing interval length. Then weights are generated in training and used in testing for  markowitz, hierarchical risk parity and black litterman algorithms.
Results for each interval, both in training and testing, are saved to compare the different algorithms. Results for same allocation and index such as sp500 are also saved.