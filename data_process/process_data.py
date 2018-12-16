import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


train_file = ''
test_file = ''
submission_file = ''
COMMENT = 'comment_text'


def process_data(train_file, test_file, submission_file):
    '''

    :param train_file:
    :param test_file:
    :param submission_file:
    :return:
    '''
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    subm = pd.read_csv(submission_file)
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train['none'] = 1-train[label_cols].max(axis=1)
    train[COMMENT].fillna("unknown", inplace=True)
    test[COMMENT].fillna("unknown", inplace=True)
    return train, test


def build_matrix():

    train, test = process_data(train_file, test_file, submission_file)
    n = train.shape[0]
    vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                          min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                          smooth_idf=1, sublinear_tf=1)
    trn_term_doc = vec.fit_transform(train[COMMENT])
    test_term_doc = vec.transform(test[COMMENT])

    return trn_term_doc, test_term_doc

