# Algorithmic trading
In this project there are several scripts to download data from free financials API's such as Finhub or Yahoo Finance and use them to backtest diferent algorithmic strategies, or to apply filters with a programatic market screener.
## Algorithms
### Portfolio allocation
+ **Algorithm1** simply takes a period of time and generates an excel file with the results of different portfolios: same allocation, markowitz, hierarchical risk parity, black litterman, and several reference indices. For the black litterman portfolio, you need to specify in **general_config.yaml** the relative and absolute views of the symbols you will use .
* **Algorithm2** is the same as **algorithm1** but pairs of training and testing intervals (testing is always next to training) are generated with some randomness using a training  interval length and testing interval length. Then weights are generated in training and used in testing for  markowitz, hierarchical risk parity and black litterman algorithms.
Results for each interval, both in training and testing, are saved to compare the different algorithms. Results for same allocation and index such as sp500 are also saved.
### NLP
* **Algorithm3**. Different stock's 10-K or 10-q documents are processed using natural language processing. Then we assign sentiments to each doc and determine alpha factors for each sentiment. Alpha factors help us to predict next year or next quarter returns based on previous 10-K or 10-Q documents.
* **Algorithm4** Identify pairs trading using natural language processing on the stocks descriptions and then clustering then using then dbscan. On the pairs we use bayeasian regresion with pymc to estimate beta using a gaussian process and then a mean reversion strategy is implemented using that beta.
### Reinforcement learning
* **Algorithm5**. Script to do reinforcement learning using gym_anytrading and stable baselines.
## Requirements
* **General requirements** .
  * ```pip install -r resources/requirements.txt``` 
* **Algorithm4 (because of pymc3)**.  You should use a different environment:
  * ```pip install -r resources/algorithm4/requirements.txt``` 
  * ``` conda install m2w64-toolchain     ```
  * ``` conda install libpython ```
* **Algorithm5 (because of tensorflow version 1.15.0 to use in stable-baselines)**.
  * ```pip install -r resources/algorithm5/requirements.txt``` 
## Proyect structure
The most important folder in the project is **src/algorithms**. There are several algorithms to backtest traing strategies. Some of them use a data structure called **StockSummary** that is saved in a pickle object and has all the stock's data that is needed.

To create that structure you need to use the script **src/index/main_load_data.py**, where data is downloaded, saved in  **StocksSummary** and then in a pickle object. If it is executed several times the pickle object is updated so previuos data is not deleted, just modified or extended with new stocks.

**StocksSummary** keeps different info, but mainly: 
* **Last stock and financial info**. To use in a market screener and filter today best/worst stocks using a big range of filters.
* **A dict of Stock objects**, one for each stock, to keep different financial series: prices, future options, historic eps, historic financials, social nework sentiments etc.
### Main project folders
* **src/index**.  
  * **main_load_data**. To download data from the API's and create the **StocksSummary**.
  * **main_screener**.Screener to apply several filters on financial and stocks data using **StocksSummary**.
* **resources**.
  * **general_properties.yaml**. Properties to configure the stocks downloaded and some parameters for the algorithms.
  * **resources/secret_properties.yaml**. Here are the API's keys:
    * **alpha_vantage_key**
    * **polygon_key**
    * **finhub_key**
 created with the **index/main_load_data**.
* **src/algorithms**. Different algorithms using the stock info on the stock series.

  
