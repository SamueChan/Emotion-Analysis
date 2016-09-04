# -*- coding:utf-8 -*-

'''
Author:Samuel Chan
'''

import jieba
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer

from cleandata import clean_train, clean_test
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


def word_to_feature(raw_line, stopwords_path='/home/samuelchan/PycharmProjects/emotion-analysis/chinese_stopwords.txt'):
    # convert a raw line to a string of word
    stopwords = {}.fromkeys([line.rstrip() for line in open(stopwords_path)])
    # 1.remove non-Chinese characters
    chinese_only = raw_line
    # 2.cut words
    words_lst = jieba.cut(chinese_only)
    # 3.remove stop words
    meaniful_words = []
    for word in words_lst:
        word = word.encode('utf-8')
        if word not in stopwords:
            meaniful_words.append(word)
    return ' '.join(meaniful_words)


def drawfeature(TRAIN_DATA_PATH='/home/samuelchan/PycharmProjects/emotion-analysis/train',
                train_filename='thetrain.csv', TEST_DATA_PATH='/home/samuelchan/PycharmProjects/emotion-analysis/test',
                test_filename='thetest.csv'):
    train_file = os.path.join(TRAIN_DATA_PATH, train_filename)
    train_data = pd.read_csv(train_file)
    n_data_train = train_data['text'].size

    test_file = os.path.join(TEST_DATA_PATH, test_filename)
    test_data = pd.read_csv(test_file)
    n_data_test = test_data['text'].size

    # # bag of words model + tfidf
    # vectorizer = CountVectorizer(analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
    # transformer = TfidfTransformer()

    # bigram + tf
    vectorizer = HashingVectorizer(ngram_range=(2, 2), non_negative=True)

    # train
    print 'Start cut word in train data set'
    train_data_word = []
    for i in xrange(n_data_train):
        if ((i + 1) % 1000 == 0):
            print 'Drawfeatures Line %d of %d' % (i + 1, n_data_train)
        train_data_word.append(word_to_feature(train_data['text'][i]))

    # print 'Start bag of word in train data set'
    # # draw features
    # train_data_features = vectorizer.fit_transform(train_data_word)
    # # train_data_features = train_data_features.toarray()
    # print 'Start tfidf in train data set'
    # train_data_features = transformer.fit_transform(train_data_features)
    # # train_data_features = train_data_features.toarray()

    print 'Start bigram model in train data set'
    train_data_features = vectorizer.fit_transform(train_data_word)

    # test
    print 'Start cut words in test data set'
    test_data_words = []
    for i in xrange(n_data_test):
        if ((i + 1) % 1000 == 0):
            print 'Drawfeatures Line %d of %d' % (i + 1, n_data_test)
        test_data_words.append(word_to_feature(test_data['text'][i]))

    # # draw feature
    # print 'Start bag of word in test data set'
    # test_data_features = vectorizer.fit_transform(test_data_words)
    # # test_data_features = test_data_features.toarray()
    # print 'Start tfidf in test data set'
    # test_data_features = transformer.fit_transform(test_data_features)
    # # test_data_features = test_data_features.toarray()

    print 'Start bigram model in test data set'
    test_data_features = vectorizer.fit_transform(test_data_words)

    # random forest
    print 'random forest'
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, train_data['label'])
    pred = forest.predict(test_data_features)
    pred = pd.Series(pred, name='TARGET')
    # pred.to_csv('BOW_TFIDF_RF5.csv', index=None, header=True)
    pred.to_csv('BI_W2V_RF1.csv', index=None, header=True)

    # multinomial navive bayes
    print 'Multinomial navive bayes'
    mnb = MultinomialNB(alpha=0.01)
    mnb = mnb.fit(train_data_features, train_data['label'])
    pred = mnb.predict(test_data_features)
    pred = pd.Series(pred, name='TARGET')
    # pred.to_csv('BOW_TFIDF_MNB5.csv', index=None, header=True)
    pred.to_csv('BI_W2V_MNB1.csv', index=None, header=True)

    # # KNN
    # print 'KNN'
    # knn = KNeighborsClassifier()
    # knn = knn.fit(train_data_features, train_data['label'])
    # pred = knn.predict(test_data_features)
    # pred = pd.Series(pred, name='TARGET')
    # pred.to_csv('BOW_TFIDF_KNN2.csv', index=None, header=True)

    # SVM
    print 'SVM'
    svm = SVC(kernel='linear')
    svm = svm.fit(train_data_features, train_data['label'])
    pred = svm.predict(test_data_features)
    pred = pd.Series(pred, name='TARGET')
    # pred.to_csv('BOW_TFIDF_SVM5.csv', index=None, header=True)
    pred.to_csv('BI_W2V_SVM1.csv', index=None, header=True)

    # GBDT
    print 'GBDT'
    gbdt = GradientBoostingClassifier()
    gbdt = gbdt.fit(train_data_features, train_data['label'])
    pred = gbdt.predict(test_data_features)
    pred = pd.Series(pred, name='TARGET')
    pred.to_csv('BI_W2V_GBDT1.csv', index=None, header=True)


# clean_train()
# clean_test()
drawfeature()
