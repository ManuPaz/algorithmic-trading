
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.express as px
import pandas as pd
import plotly.express as px
import pandas as pd
def plot_returns(returns,**kwargs):
    fig=plt.figure()
    plt.plot(returns.index,returns,color="green")
    plt.title(kwargs["title"])
    plt.show()
    path="reports/plots/returns"
    if not os.path.isdir(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path,kwargs["title"]))


def plot_prices(df,column):
    df["date"]=df.index
    fig = px.line(df, x='date', y=column)

    fig.show()
    df = df.drop("date", axis=1)