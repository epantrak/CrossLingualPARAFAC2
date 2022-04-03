import os
import re
import jieba
import pandas as pd
from fugashi import Tagger

import json
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.snowball import FrenchStemmer
from nltk.stem.snowball import GermanStemmer
from nltk.stem.snowball import ItalianStemmer
from nltk.stem.snowball import RussianStemmer
from nltk.stem.snowball import SpanishStemmer

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from copy import deepcopy

tagger = Tagger('-Owakati -b 5000')

fr_stemmer = FrenchStemmer()
en_stemmer = EnglishStemmer()
de_stemmer = GermanStemmer()
it_stemmer = ItalianStemmer()
ru_stemmer = RussianStemmer()
es_stemmer = SpanishStemmer()

with open('Datasets/ja_stop_words.json') as data_file:
    ja_stop_words = json.load(data_file)

with open('Datasets/zh_stop_words.json') as data_file:
    zh_stop_words = json.load(data_file)


def read_aligned_raw_data(data_folder, language, masked=True):
    if masked:
        file_train = os.path.join(data_folder, language + '_masked.txt')
    else:
        file_train = os.path.join(data_folder, language + '.txt')

    f = open(file_train, 'r', encoding="UTF-8")
    data = f.readlines()
    data = [x.rstrip('\n') for x in data]
    df_train = pd.DataFrame(data, columns=['Text'])
    return df_train


def en_tokenize(text):
    tokens = [word for word in nltk.word_tokenize(text, language='english') if word.isalpha()]
    stems = [en_stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
    return stems


def fr_tokenize(text):
    tokens = [word for word in nltk.word_tokenize(text, language='french') if word.isalpha()]
    stems = [fr_stemmer.stem(t) for t in tokens if t not in stopwords.words('french')]
    return stems


def de_tokenize(text):
    tokens = [word for word in nltk.word_tokenize(text, language='german') if word.isalpha()]
    stems = [de_stemmer.stem(t) for t in tokens if t not in stopwords.words('german')]
    return stems


def it_tokenize(text):
    tokens = [word for word in nltk.word_tokenize(text, language='italian') if word.isalpha()]
    stems = [it_stemmer.stem(t) for t in tokens if t not in stopwords.words('italian')]
    return stems


def es_tokenize(text):
    tokens = [word for word in nltk.word_tokenize(text, language='spanish') if word.isalpha()]
    stems = [es_stemmer.stem(t) for t in tokens if t not in stopwords.words('spanish')]
    return stems


def ru_tokenize(text):
    tokens = [word for word in nltk.word_tokenize(text, language='russian') if word.isalpha()]
    stems = [ru_stemmer.stem(t) for t in tokens if t not in stopwords.words('russian')]
    return stems


def zh_tokenize(text):
    tokens = ' '.join(jieba.lcut(text)).split()
    stop_words = zh_stop_words + stopwords.words('english')
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens


def ja_tokenize(text):
    text = re.sub(r"[0-9]+", '', re.sub(r"\W", ' ', text))
    tokens = [word.surface for word in tagger(text)]
    stop_words = ja_stop_words + stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


def tokenize_data(language, df_train, df_dev=None, df_test=None, vocab=None, ngram_range=(1, 1),
                  cased=True, use_tfidf=True, norm='l2', tok_lib='nltk'):
    if language == 'english':
        tokenizer = en_tokenize
    elif language == 'french':
        tokenizer = fr_tokenize
    elif language == 'german':
        tokenizer = de_tokenize
    elif language == 'italian':
        tokenizer = it_tokenize
    elif language == 'spanish':
        tokenizer = es_tokenize
    elif language == 'russian':
        tokenizer = ru_tokenize
    elif language == 'chinese':
        tokenizer = zh_tokenize
    elif language == 'japanese':
        tokenizer = ja_tokenize
    else:
        raise ValueError('Select correct language')

    tokenized_train, tokenized_dev, tokenized_test = None, None, None

    vectorizer = CountVectorizer(tokenizer=lambda text: tokenizer(text),
                                 ngram_range=ngram_range,
                                 lowercase=not cased,
                                 vocabulary=vocab,
                                 binary=False)

    tokenized_train = vectorizer.fit_transform(df_train['Text'].values.tolist())

    print('Task dataset {} vocab size {}'.format(language, len(vectorizer.vocabulary_)))

    if df_dev is not None:
        tokenized_dev = vectorizer.transform(df_dev['Text'].values.tolist())
    if df_test is not None:
        tokenized_test = vectorizer.transform(df_test['Text'].values.tolist())

    if use_tfidf:
        tfidf_transformer = TfidfTransformer(use_idf=True, norm=norm)
        tokenized_train = tfidf_transformer.fit_transform(tokenized_train)
        if df_dev is not None:
            tokenized_dev = tfidf_transformer.transform(tokenized_dev)
        if df_test is not None:
            tokenized_test = tfidf_transformer.transform(tokenized_test)
    else:
        tokenized_train = preprocessing.normalize(tokenized_train, norm=norm, axis=1)
        if df_dev is not None:
            tokenized_dev = preprocessing.normalize(tokenized_dev, norm=norm, axis=1)
        if df_test is not None:
            tokenized_test = preprocessing.normalize(tokenized_test, norm=norm, axis=1)

    return tokenized_train, tokenized_dev, tokenized_test, vectorizer


def read_raw_data_mldoc(data_folder, num_train, language):
    file_train = os.path.join(data_folder, language.lower() + ".train." + str(num_train))
    file_dev = os.path.join(data_folder, language.lower() + ".dev")
    file_test = os.path.join(data_folder, language.lower() + ".test")

    df_train = pd.read_csv(file_train, delimiter='\t', encoding="UTF-8", names=['Label', 'Text'])
    df_dev = pd.read_csv(file_dev, delimiter='\t', encoding="UTF-8", names=['Label', 'Text'])
    df_test = pd.read_csv(file_test, delimiter='\t', encoding="UTF-8", names=['Label', 'Text'])

    return df_train, df_dev, df_test


def encode_labels(df_train, df_dev=None, df_test=None):
    labels_train, labels_dev, labels_test = None, None, None
    # Create label matrices for train and test
    le = preprocessing.LabelEncoder()
    labels_train = le.fit_transform(df_train["Label"].tolist())

    if df_dev is not None:
        labels_dev = le.transform(df_dev["Label"].tolist())
    if df_test is not None:
        labels_test = le.transform(df_test["Label"].tolist())

    return labels_train, labels_dev, labels_test, le.classes_


def read_raw_data_amazon(data_folder, language, domain):
    file_train = os.path.join(data_folder, language.lower(), domain, "train.review")
    file_dev = os.path.join(data_folder, language.lower(), domain, "dev.review")
    file_test = os.path.join(data_folder, language.lower(), domain, "test.review")

    df_train = pd.read_csv(file_train, delimiter='\t', encoding="UTF-8", names=['Label', 'Text'])
    df_dev = pd.read_csv(file_dev, delimiter='\t', encoding="UTF-8", names=['Label', 'Text'])
    df_test = pd.read_csv(file_test, delimiter='\t', encoding="UTF-8", names=['Label', 'Text'])

    return df_train, df_dev, df_test


def compute_class(X_tr, X_dev, X_tst, lb_tr, lb_dev, lb_tst):

    print("Fitting the classifier to the training set")
    param_grid = {"C": np.logspace(-5, 10, 20)}
    clf = LogisticRegression(solver='lbfgs', max_iter=20000, n_jobs=-1)

    # Create a classifier copy for parameter selection
    clf_search = deepcopy(clf)

    X_combined = np.vstack([X_tr, X_dev])
    lb_combined = np.hstack([lb_tr, lb_dev])

    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = np.hstack((np.ones(X_tr.shape[0], dtype=int) * -1, np.zeros(X_dev.shape[0], dtype=int)))
    ps = PredefinedSplit(split_index)

    # Optimize parameters using dev set
    search = GridSearchCV(clf_search, param_grid=param_grid, cv=ps,
                          n_jobs=-1, verbose=1, refit=False).fit(X_combined, lb_combined)

    print(search.best_params_)

    clf.set_params(**search.best_params_).fit(X_tr, lb_tr)
    accuracies = {}
    for X_data, ld_data, type_data in zip([X_tst, X_dev], [lb_tst, lb_dev], ['test', 'dev']):
        y_pred = clf.predict(X_data)
        acc = accuracy_score(ld_data, y_pred)
        accuracies.update({type_data: acc})
        print('Classification {} accuracy {}'.format(type_data, acc))

    return accuracies
