# -*- coding:utf-8 -*-

'''

This file contain code to clean the raw data and save the data as csv format.
For training set, the first column represents the label; the second column represents the text.
For test set ,the first column represents the text.
'''

import os
import re
import pandas as pd


def clean_train(TRAIN_DATA_PATH='/home/samuelchan/PycharmProjects/emotion-analysis/train',
                out_train_filename='thetrain.csv'):
    print 'start cleaning train data'
    train_file_names = os.listdir(TRAIN_DATA_PATH)
    train_data_list = []
    for train_file_name in train_file_names:
        if not train_file_name.endswith('.txt'):
            continue
        train_file = os.path.join(TRAIN_DATA_PATH, train_file_name)

        # draw sentiment label
        label = int(train_file_name[0])
        print label

        with open(train_file, 'r') as f:
            lines = f.read().splitlines()

        labels = [label] * len(lines)

        labels_series = pd.Series(labels)
        lines_series = pd.Series(lines)

        # construct dataframe
        data_pd = pd.concat([labels_series, lines_series], axis=1)
        train_data_list.append(data_pd)

        train_data_pd = pd.concat(train_data_list, axis=0)

        # output train data
        train_data_pd.columns = ['label', 'text']
        train_data_pd.to_csv(os.path.join(TRAIN_DATA_PATH, out_train_filename), index=None,
                             encoding='utf-8', header=True)


def clean_test(TEST_DATA_PATH='/home/samuelchan/PycharmProjects/emotion-analysis/test',
               test_filename='Emotion-Analysis-dataset_test1.csv', out_test_filename='thetest.csv'):
    print 'start cleaning test data'
    test_file = os.path.join(TEST_DATA_PATH, test_filename)

    with open(test_file, 'r') as f:
        lines = f.read().splitlines()

    lines_series = pd.Series(lines)

    test_data_list = pd.Series(lines_series, name='text')

    # output test data
    test_data_list.to_csv(os.path.join(TEST_DATA_PATH, out_test_filename), index=None, encoding='utf-8', header=True)


