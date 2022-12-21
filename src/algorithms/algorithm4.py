import re
import os
import numpy as np
if os.getcwd().split("\\")[-1] == "algorithms":
    os.chdir("../../")
import logging.config
import src.functions.nlp as nlp

logging.config.fileConfig('resources/logging.conf')
logger = logging.getLogger('general')
import src.utils.load_config as load_config
import datetime
import pickle
from wordcloud import WordCloud
import src.functions.clustering as clustering
import src.functions.machine_learning as machine_learning
import pandas as pd
general_config = load_config.general_config()
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


def getStrategyPortfolioWeights(rolling_beta, stock_name1, stock_name2, data, smoothing_window=15):
    data1 = data[stock_name1].ffill().fillna(0).values


    data2 = data[stock_name2].ffill().fillna(0).values
    # initial signal rebalance
    fixed_beta = rolling_beta[smoothing_window]
    signal = fixed_beta * data1 - data2
    smoothed_signal = pd.Series(signal).rolling(smoothing_window).mean()
    d_smoothed_signal = smoothed_signal.diff()
    trading = "not"
    trading_start = 0
    leverage = 0 * data.copy()
    for i in range(smoothing_window, data1.shape[0]):
        leverage.iloc[i, :] = leverage.iloc[i - 1, :]
        if trading == "not":
            # dynamically rebalance the signal when not trading
            fixed_beta = rolling_beta[i]
            signal = fixed_beta * data1 - data2
            smoothed_signal = pd.Series(signal).rolling(smoothing_window).mean()
            d_smoothed_signal = smoothed_signal.diff()
            if smoothed_signal[i] > 0 and d_smoothed_signal[i] < 0:
                    leverage.iloc[i, 0] = -fixed_beta / (abs(fixed_beta) + 1)
                    leverage.iloc[i, 1] = 1 / (abs(fixed_beta) + 1)
                    trading = "short"
                    trading_start = smoothed_signal[i]
            elif smoothed_signal[i] < 0 and d_smoothed_signal[i] > 0:
                fixed_beta = rolling_beta[i]
                leverage.iloc[i, 0] = fixed_beta / (abs(fixed_beta) + 1)
                leverage.iloc[i, 1] = -1 / (abs(fixed_beta) + 1)
                trading = "long"
                trading_start = smoothed_signal[i]
            else:
                leverage.iloc[i, 0] = 0
                leverage.iloc[i, 1] = 0
        elif trading == "long":
            # a failed trade
            if smoothed_signal[i] < trading_start:
                leverage.iloc[i, 0] = 0
                leverage.iloc[i, 1] = 0
                trading = "not"
            # a successful trade
            if smoothed_signal[i] > 0:
                leverage.iloc[i, 0] = 0
                leverage.iloc[i, 1] = 0
                trading = "not"
        elif trading == "short":
            # a failed trade
            if smoothed_signal[i] > trading_start:
                leverage.iloc[i, 0] = 0
                leverage.iloc[i, 1] = 0
                trading = "not"
            # a successful trade
            if smoothed_signal[i] < 0:
                leverage.iloc[i, 0] = 0
                leverage.iloc[i, 1] = 0
                trading = "not"

    return leverage





def backtest(pricingDF, leverageDF, start_cash):
    """Backtests pricing based on some given set of leverage. Leverage works such that it happens "overnight",
    so leverage for "today" is applied to yesterday's close price. This algo can handle NaNs in pricing data
    before a stock exists, but ffill() should be used for NaNs that occur after the stock has existed, even
    if that stock ceases to exist later."""

    pricing = pricingDF.values
    leverage = leverageDF.values

    shares = np.zeros_like(pricing)
    cash = np.zeros(pricing.shape[0])
    cash[0] = start_cash
    curr_price = np.zeros(pricing.shape[1])
    curr_price_div = np.zeros(pricing.shape[1])

    for t in range(1, pricing.shape[0]):

        if np.any(leverage[t] != leverage[t - 1]):
            # handle non-existent stock values
            curr_price[:] = pricing[t - 1]  # you can multiply with this one
            curr_price[np.isnan(curr_price)] = 0
            trading_allowed = (curr_price != 0)
            curr_price_div[:] = curr_price  # you can divide with this one
            curr_price_div[~trading_allowed] = 1

            # determine new positions (warning: leverage to non-trading_allowed stocks is just lost)
            portfolio_value = (shares[t - 1] * curr_price).sum() + cash[t - 1]
            target_shares = trading_allowed * (portfolio_value * leverage[t]) // curr_price_div

            # rebalance
            shares[t] = target_shares
            cash[t] = cash[t - 1] - ((shares[t] - shares[t - 1]) * curr_price).sum()

        else:

            # maintain positions
            shares[t] = shares[t - 1]
            cash[t] = cash[t - 1]

    returns = (shares * np.nan_to_num(pricing)).sum(axis=1) + cash
    pct_returns = (returns - start_cash) / start_cash
    return (
        pd.DataFrame(shares, index=pricingDF.index, columns=pricingDF.columns),
        pd.Series(cash, index=pricingDF.index),
        pd.Series(pct_returns, index=pricingDF.index)
    )



def visualize_cluster(profiles_df, cluster, ):
    num_stocks = len(list(profiles_df[profiles_df.cluster == cluster].index)[:])
    num_cols = 2
    num_rows = num_stocks // num_cols
    num_rows += num_stocks % num_cols
    position = range(1, num_stocks + 1)
    fig = plt.figure(1)
    fig.set_figheight(10)
    fig.set_figwidth(16)
    for index in range(len(list(profiles_df[profiles_df.cluster == cluster].index)[:])):
        wordcloud_ticker = list(profiles_df[profiles_df.cluster == cluster].index)[index]
        wordcloud_profile = [profiles_df.loc[profiles_df["cluster"] == cluster]["longBusinessSummary"][index]]
        vect = CountVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            strip_accents='unicode',
            max_features=3000,  # we limit the generation of tokens to the top 3000
            stop_words=nlp.get_stop_words()
        )
        wordcloud_X = vect.fit_transform(wordcloud_profile)
        wordcloud_word_features = list(zip(wordcloud_X.toarray()[0], vect.get_feature_names_out()))
        wordcloud_frequencies = []
        for word in wordcloud_word_features[:]:
            if word[0] > 0:
                new_word = (word[1], word[0])
                wordcloud_frequencies.append(new_word)
        wordcloud = WordCloud(background_color="white", max_font_size=48).generate_from_frequencies(
            dict(wordcloud_frequencies))
        fig.add_subplot(num_rows, num_cols, position[index])

        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title(wordcloud_ticker)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with open("resources/stocks_summary.obj", "rb") as stocks_summary_file:
        stocks_summary = pickle.load(stocks_summary_file)

    profiles = stocks_summary.stock_profile.loc[:, ["country", "industry", "sector", "longBusinessSummary"]]
    missing_text = '""'
    profiles = profiles.loc[~(profiles["longBusinessSummary"] == missing_text)]
    profiles = profiles.droplevel(1)
    profiles = profiles[~profiles.index.duplicated(keep='first')]
    profiles = profiles.dropna(subset=["longBusinessSummary"])
    df_words, X = nlp.get_words_count(profiles, "longBusinessSummary")
    # machine_learning.bayesian_cointegration_regresion(data1, data2)
    clustered_series = clustering.dbscan(X, profiles.index, eps=1.05, min_samples=2)
    pair_clusters = clustered_series.value_counts()[clustered_series.value_counts() < 3].index.values
    print(pair_clusters)
    print("\nTotal pair clusters discovered: %d" % len(pair_clusters))
    cluster = clustered_series['TENB']
    profiles = profiles.loc[clustered_series.index].reindex(clustered_series.index)
    profiles["cluster"] = clustered_series
    prices = stocks_summary.get_all_prices_one_column(list(stocks_summary.stock_objects.keys()), "adj_close")
    initial_date = datetime.datetime.strptime(general_config["algorithm4"]["initial_date"], "%Y-%m-%d")
    prices = prices.loc[initial_date:]
    cluster = profiles.loc["RFP","cluster"]
    clustering.plot_cluster(profiles, prices, cluster,)
    clustering.plot_cluster(profiles, prices, cluster, plot_mean=True)
    visualize_cluster(profiles, cluster)

    for cluster in pair_clusters:
        clustered_index = profiles.loc[profiles["cluster"] == cluster].index
        print(cluster,clustered_index)
    clustered_index = profiles.loc[profiles["cluster"] == cluster].index
    prices_cluster = prices.loc[:, clustered_index]
    prices_cluster = prices_cluster.diff().cumsum().bfill()
    rolling_beta=machine_learning.bayesian_cointegration_regresion(prices_cluster.iloc[:, 0].values,
                                                      prices_cluster.iloc[:, 1].values, prices_cluster.index)
    portfolioWeights = getStrategyPortfolioWeights(rolling_beta,  prices_cluster.columns[0],  prices_cluster.columns[1], prices_cluster).fillna(0)
    shares, cash, returns = backtest(prices_cluster, portfolioWeights, 1e6)
    plt.figure(figsize=(18, 8))
    ax = plt.gca()
    plt.title("Return Profile of Algorithm")
    plt.ylabel("Percent Returns")
    returns.plot(ax=ax, linewidth=3)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.show()
