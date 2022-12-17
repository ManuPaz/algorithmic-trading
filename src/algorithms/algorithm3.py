
import datetime
import os
import json
import pandas as pd
from src.functions import metrics

os.chdir("../../")
from src.utils import save_results
import numpy as np
import logging.config
from src.functions.financial_functions import portfolio_returns
logging.config.fileConfig('resources/logging.conf')
logger = logging.getLogger('general')
import pickle
import nltk
if __name__=="__main__":
    with open("resources/stocks_summary.obj", "rb") as stocks_summary_file:
        stocks_summary = pickle.load(stocks_summary_file)

    with open("resources/other_equities.obj", "rb") as other_equities_file:
        other_equities = pickle.load(other_equities_file)

    stocks_summary
