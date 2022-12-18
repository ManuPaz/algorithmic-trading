import re
import os
from tqdm import tqdm
import numpy as np
os.chdir("../../")
import logging.config
from src.functions.apis import SecAPI
import src.functions.nlp as nlp
import alphalens as al
logging.config.fileConfig('resources/logging.conf')
logger = logging.getLogger('general')
import src.utils.load_config as load_config
import pickle
import pandas as pd
general_config = load_config.general_config()
import matplotlib.pyplot as plt
import pprint
def process_filings(cik_lookup):
    sec_api = SecAPI()
    sec_data = {}

    for ticker in cik_lookup.keys():
        sec_data[ticker] = sec_api.get_sec_data(cik=ticker, doc_type="10-K")
    raw_fillings_by_ticker = {}
    for ticker, data in sec_data.items():
        raw_fillings_by_ticker[ticker] = {}
        for index_url, file_type, file_date in tqdm(data, desc='Downloading {} Fillings'.format(ticker),
                                                    unit='filling'):
            if (file_type == '10-K'):
                file_url = index_url.replace('-index.htm', '.txt').replace('.txtl', '.txt')

                raw_fillings_by_ticker[ticker][file_date] = sec_api.get(file_url)
    filling_documents_by_ticker = {}
    for ticker, raw_fillings in raw_fillings_by_ticker.items():
        filling_documents_by_ticker[ticker] = {}
        for file_date, filling in tqdm(raw_fillings.items(), desc='Getting Documents from {} Fillings'.format(ticker),
                                       unit='filling'):
            filling_documents_by_ticker[ticker][file_date] = sec_api.get_documents(filling)
    filling_documents_by_ticker
    ten_ks_by_ticker = {}
    for ticker, filling_documents in filling_documents_by_ticker.items():
        ten_ks_by_ticker[ticker] = []
        for file_date, documents in filling_documents.items():
            for document in documents:
                if sec_api.get_document_type(document) == '10-k':
                    ten_ks_by_ticker[ticker].append({
                        'cik': cik_lookup[ticker],
                        'file': document,
                        'file_date': file_date})

    for ticker, ten_ks in ten_ks_by_ticker.items():
        for ten_k in tqdm(ten_ks, desc='Cleaning {} 10-Ks'.format(ticker), unit='10-K'):
            ten_k['file_clean'] = nlp.clean_text(ten_k['file'])

    word_pattern = re.compile('\w+')
    for ticker, ten_ks in ten_ks_by_ticker.items():
        for ten_k in tqdm(ten_ks, desc='Lemmatize {} 10-Ks'.format(ticker), unit='10-K'):
            ten_k['file_lemma'] = nlp.lemmatize_words(word_pattern.findall(ten_k['file_clean']))

    lemma_english_stopwords = nlp.get_stop_words()
    for ticker, ten_ks in ten_ks_by_ticker.items():
        for ten_k in tqdm(ten_ks, desc='Remove Stop Words for {} 10-Ks'.format(ticker), unit='10-K'):
            ten_k['file_lemma'] = [word for word in ten_k['file_lemma'] if word not in lemma_english_stopwords]
    print('Stop Words Removed')
    return ten_ks_by_ticker


if __name__ == "__main__":
    with open("resources/stocks_summary.obj", "rb") as stocks_summary_file:
        stocks_summary = pickle.load(stocks_summary_file)

    cik_lookup = {
        'AMZN': '0001018724',
        'BMY': '0000014272',
        'CNP': '0001130310',
        'CVX': '0000093410',
        'FL': '0000850209',
        'FRT': '0000034903',
        'HON': '0000773840'}
    pricing = stocks_summary.get_all_prices_one_column(list(cik_lookup.keys()),"adj_close")
    example_ticker = "AMZN"
    ten_ks_file="resources/algorithm3/tens_ks.obj"
    if os.path. isfile( ten_ks_file):
        with open( ten_ks_file, "rb") as  ten_ks:
            ten_ks_by_ticker= pickle.load(ten_ks)
    else:
        ten_ks_by_ticker = process_filings(cik_lookup)
        with open(ten_ks_file, "wb") as ten_ks:
            pickle.dump(ten_ks_by_ticker,ten_ks,protocol=pickle.HIGHEST_PROTOCOL)
    for key, item in ten_ks_by_ticker.items():
        ten_ks_by_ticker[key]=list(np.flip(item))
    sentiment_df = nlp.get_sentiment_df()
    sentiments = general_config["algorithm3"]["sentiments"]
    sentiment_bow_ten_ks = {}

    for ticker, ten_ks in ten_ks_by_ticker.items():
        lemma_docs = [' '.join(ten_k['file_lemma']) for ten_k in ten_ks]

        sentiment_bow_ten_ks[ticker] = {
            sentiment: nlp.get_bag_of_words(sentiment_df[sentiment_df[sentiment]]['word'], lemma_docs)
            for sentiment in sentiments}

    sentiment_bow_ten_ks
    file_dates = {
        ticker: [ten_k['file_date'] for ten_k in ten_ks]
        for ticker, ten_ks in ten_ks_by_ticker.items()}
    jaccard_similarities = {
        ticker: {
            sentiment_name: nlp.get_jaccard_similarity(sentiment_values)
            for sentiment_name, sentiment_values in ten_k_sentiments.items()}
        for ticker, ten_k_sentiments in sentiment_bow_ten_ks.items()}

    nlp.plot_similarities(
        ([jaccard_similarities[example_ticker][sentiment] for sentiment in sentiments]),
        (file_dates[example_ticker][1:]),
        'Jaccard Similarities for {} Sentiment'.format(example_ticker),
        sentiments)

    sentiment_tfidf_ten_ks = {}
    for ticker, ten_ks in ten_ks_by_ticker.items():
        lemma_docs = [' '.join(ten_k['file_lemma']) for ten_k in ten_ks]

        sentiment_tfidf_ten_ks[ticker] = {
            sentiment: nlp.get_tfidf(sentiment_df[sentiment_df[sentiment]]['word'], lemma_docs)
            for sentiment in sentiments}

    cosine_similarities = {
        ticker: {
            sentiment_name: nlp.get_cosine_similarity(sentiment_values)
            for sentiment_name, sentiment_values in ten_k_sentiments.items()}
        for ticker, ten_k_sentiments in sentiment_tfidf_ten_ks.items()}
    nlp.plot_similarities(
        [cosine_similarities[example_ticker][sentiment] for sentiment in sentiments],
        file_dates[example_ticker][1:],
        'Cosine Similarities for {} Sentiment'.format(example_ticker),
        sentiments)

    cosine_similarities_df_dict = {'date': [], 'ticker': [], 'sentiment': [], 'value': []}
    for ticker, ten_k_sentiments in cosine_similarities.items():
        for sentiment_name, sentiment_values in ten_k_sentiments.items():
            for sentiment_values, sentiment_value in enumerate(sentiment_values):
                cosine_similarities_df_dict['ticker'].append(ticker)
                cosine_similarities_df_dict['sentiment'].append(sentiment_name)
                cosine_similarities_df_dict['value'].append(sentiment_value)
                cosine_similarities_df_dict['date'].append(file_dates[ticker][1:][sentiment_values])
    cosine_similarities_df = pd.DataFrame(cosine_similarities_df_dict)
    cosine_similarities_df['date'] = pd.DatetimeIndex(cosine_similarities_df['date']).year
    cosine_similarities_df['date'] = pd.to_datetime(cosine_similarities_df['date'], format='%Y')
    factor_data = {}
    skipped_sentiments = []
    pricing=pricing.resample("y").last()
    pricing.index=pd.DatetimeIndex(pricing.index).year
    pricing.index=pd.to_datetime( pricing.index, format='%Y')

    #####
    for sentiment in sentiments:
        cs_df = cosine_similarities_df[(cosine_similarities_df['sentiment'] == sentiment)]

        cs_df = cs_df.pivot(index='date', columns='ticker', values='value')
        cs_df = cs_df.loc[pricing.index[0]:]
        try:
            data = al.utils.get_clean_factor_and_forward_returns(cs_df.stack(), pricing.loc[cs_df.index], quantiles=5,
                                                                 bins=None, periods=[1])
            factor_data[sentiment] = data
        except:
            skipped_sentiments.append(sentiment)
    if skipped_sentiments:
        print('\nSkipped the following sentiments:\n{}'.format('\n'.join(skipped_sentiments)))
    unixt_factor_data = {
        factor: data.set_index(pd.MultiIndex.from_tuples(
            [(x.timestamp(), y) for x, y in data.index.values],
            names=['date', 'asset']))
        for factor, data in factor_data.items()}
    ls_factor_returns = pd.DataFrame()
    plt.figure()
    for factor_name, data in factor_data.items():
        ls_factor_returns[factor_name] = al.performance.factor_returns(data).iloc[:, 0]
        plt.plot((1 + ls_factor_returns).cumprod())
    plt.show()

    ls_FRA = pd.DataFrame()
    plt.figure()
    for factor, data in unixt_factor_data.items():
        ls_FRA[factor] = al.performance.factor_rank_autocorrelation(data)
        plt.plot(ls_FRA)
    plt.title("Factor Rank Autocorrelation")
    plt.show()

    daily_annualization_factor = np.sqrt(252)
    sharpe=(daily_annualization_factor * ls_factor_returns.mean() / ls_factor_returns.std()).round(2)
    pprint.pprint(sharpe)