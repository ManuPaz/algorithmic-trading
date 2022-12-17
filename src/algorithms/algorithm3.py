
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
from bs4 import BeautifulSoup
import nltk
if __name__=="__main__":
    nltk.download('stopwords')
    nltk.download('wordnet')
