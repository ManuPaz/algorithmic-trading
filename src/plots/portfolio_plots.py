
import seaborn as sns
import matplotlib.pyplot as plt
import os
def plot_returns(returns,**kwargs):
    fig=plt.figure()
    plt.plot(returns.index,returns,color="green")
    plt.title(kwargs["title"])
    plt.show()
    path="reports/plots/returns"
    if not os.path.isdir(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path,kwargs["title"]))
