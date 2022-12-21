
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def dbscan(x,indice,eps=1.05, min_samples=2):
    clf = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clf.fit_predict(x)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("\nTotal custers discovered: %d" % n_clusters_)
    clustered = labels
    clustered_series = pd.Series(index=indice, data=clustered.flatten())
    clustered_series = clustered_series[clustered_series != -1]
    plt.figure(figsize=(16, 8))
    plt.barh(
        range(len(clustered_series.value_counts())),
        clustered_series.value_counts(),
        alpha=0.625
    )
    plt.title('Cluster Member Counts')
    plt.xlabel('Stocks in Cluster');
    plt.tight_layout()
    plt.show()
    return clustered_series


def plot_cluster(df_clusters,pricing,which_cluster, plot_mean=False):
    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    symbols = list(df_clusters[df_clusters["cluster"] == which_cluster].index)

    pricing = pricing[symbols]
    means = np.log(pricing).mean()
    data = np.log(pricing).sub(means)

    if plot_mean:
        means = data.mean(axis=1).rolling(window=21).mean().shift(1)
        data.sub(means, axis=0).plot()
        plt.axhline(0, lw=3, ls='--', label='mean', color='k')
        plt.legend(loc=0)
    else:
        data.plot()


    plt.tight_layout()
    plt.show()



