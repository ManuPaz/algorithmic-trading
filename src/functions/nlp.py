from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
nltk.download('stopwords')
nltk.download('wordnet')
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import src.utils.load_config as load_config
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
general_config = load_config.general_config()
def get_bag_of_words(sentiment_words, docs):
    vec = CountVectorizer(vocabulary=sentiment_words)
    vectors = vec.fit_transform(docs)
    words_list = vec.get_feature_names_out()
    bag_of_words = np.zeros([len(docs), len(words_list)])

    for i in range(len(docs)):
        bag_of_words[i] = vectors[i].toarray()[0]
    return bag_of_words.astype(int)

def preprocess_text(text):

    text = re.sub(r'\d+', ' ', text)

    return text
def get_words_count(df,column):
    vect = CountVectorizer(stop_words='english')
    stop_words = list(vect.get_stop_words())
    vect = CountVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        strip_accents='unicode',
        max_features=3000,  # we limit the generation of tokens to the top 3000
        stop_words=stop_words,
        preprocessor=preprocess_text
    )
    X = vect.fit_transform(df[column])
    tfidf = TfidfTransformer()
    X_idf = tfidf.fit_transform(X)
    X_idf
    return vect.get_feature_names_out(),X_idf
def get_sentiment_df():
    sentiments = general_config["algorithm3"]["sentiments"]

    sentiment_df = pd.read_csv(general_config["loughran_mcdonald"])
    sentiment_df.columns = [column.lower() for column in sentiment_df.columns]  # Lowercase the columns for ease of use

    # Remove unused information
    sentiment_df = sentiment_df[sentiments + ['word']]
    sentiment_df[sentiments] = sentiment_df[sentiments].astype(bool)
    sentiment_df = sentiment_df[(sentiment_df[sentiments]).any(1)]

    # Apply the same preprocessing to these words as the 10-k words
    sentiment_df['word'] = lemmatize_words(sentiment_df['word'].str.lower())
    sentiment_df = sentiment_df.drop_duplicates('word')

    print(sentiment_df.head())
    return sentiment_df
def remove_html_tags(text):
    text = BeautifulSoup(text, 'html.parser').get_text()

    return text
def clean_text(text):
    text = text.lower()
    text = remove_html_tags(text)

    return text
def get_stop_words():
    return lemmatize_words(stopwords.words('english'))

def lemmatize_words(words):
    lemmatized_words = [WordNetLemmatizer().lemmatize(word, 'v') for word in words]

    return lemmatized_words


def get_jaccard_similarity(bag_of_words_matrix,):
    jaccard_similarities = []
    bag_of_words_matrix = np.array(bag_of_words_matrix, dtype=bool)

    for i in range(len(bag_of_words_matrix) - 1):
        u = bag_of_words_matrix[i]
        v = bag_of_words_matrix[i + 1]
        jaccard_similarities.append(jaccard_score(u, v,))

    return jaccard_similarities

def plot_similarities(similarities_list, dates, title, labels):
    assert len(similarities_list) == len(labels)

    plt.figure(1, figsize=(10, 7))
    for similarities, label in zip(similarities_list, labels):
        plt.title(title)
        plt.plot(dates, similarities, label=label)
        plt.legend()
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()





def get_tfidf(sentiment_words, docs):
    vec = TfidfVectorizer(vocabulary=sentiment_words)
    tfidf = vec.fit_transform(docs)

    return tfidf.toarray()


def get_cosine_similarity(tfidf_matrix):
    cosine_similarities = []

    for i in range(len(tfidf_matrix) - 1):
        cosine_similarities.append(cosine_similarity(tfidf_matrix[i].reshape(1, -1), tfidf_matrix[i + 1].reshape(1, -1))[0, 0])

    return cosine_similarities
