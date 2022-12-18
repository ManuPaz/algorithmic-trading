import re
import os

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

general_config = load_config.general_config()
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


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
    machine_learning.bayesian_cointegration_regresion(prices_cluster.iloc[:, 0].values,
                                                      prices_cluster.iloc[:, 1].values, prices_cluster.index)
