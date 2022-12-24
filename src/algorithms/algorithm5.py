


import os
os.chdir("../..")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import  tensorflow as tf
print(tf.test.is_gpu_available(
    cuda_only=True ,
))
import pickle
import gym
import gym_anytrading
from src.plots import portfolio_plots
from stable_baselines.common.vec_env import  DummyVecEnv
from stable_baselines import  A2C
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
window_size=5
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions


def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['close', 'open', 'high', 'low']].to_numpy()[start:end]
    return prices, signal_features


class MyEnv(StocksEnv):
    _process_data = my_process_data
custom_env=False

if __name__ == "__main__":
    r_learning_file="resources/algorithm5/r_learnings.obj"
    with open("resources/algorithm5/stocks_summary.obj", "rb") as stocks_summary_file:
        stocks_summary = pickle.load(stocks_summary_file)
    with open("resources/algorithm5/other_equities.obj", "rb") as other_equities_file:
        other_equities = pickle.load(other_equities_file)
    if os.path.isfile(r_learning_file):
        with open(r_learning_file, "rb") as r_file:
            model= pickle.load(r_file)
    index= other_equities.index.loc[:,["nasdaq100"]]
    prices=stocks_summary.stock_objects["GOOG"].prices
    low_interval = 10
    high_interval = 100
    index=index.transform(lambda x:np.log2(x))
    #portfolio_plots.plot_prices(index,'nasdaq100')
    factor=prices["adj_close"]/prices["close"]
    prices.loc[:,["high","low","open"]]=factor.values.reshape(-1,1) *prices.loc[:,["high","low","open"]]
    prices["Close"]=prices["adj_close"]




    if custom_env:
        env_maker=lambda: MyEnv(df=prices, frame_bound=(low_interval,high_interval), window_size=window_size)
    else:
        env_maker = lambda: gym.make("stocks-v0", df=prices, frame_bound=(low_interval, high_interval), window_size=window_size)

    env=DummyVecEnv([env_maker])

    model=A2C("MlpLstmPolicy",env,verbose=2)
    model.learn(total_timesteps=100000, )
    #with open(r_learning_file, "wb") as rf_file:
      #  pickle.dump(model,rf_file)


    if custom_env:
        env=MyEnv(df=prices, frame_bound=(low_interval, high_interval), window_size=window_size)
    else:
        env = gym.make("stocks-v0", df=prices, frame_bound=(low_interval, high_interval), window_size=window_size)

    obs=env.reset()
    while True:
        obs=obs[np.newaxis,...]
        action,_states=model.predict(obs)
        obs,rewards,done,info=env.step(action)
        if done:
            print("info",info)
            break
    plt.figure()
    env.render_all()
    plt.show()