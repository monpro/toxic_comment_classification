from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import re
import numpy as np, pandas as pd
import data_process.process_data as dp

train_file = ''
test_file = ''
submission_file = ''


def tokenize(s):
    pattern = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
    return pattern.sub(r' \1 ', s).split()


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        self._r = None
        self._clf = None

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def pr(self, x, y_i, y):
        p = x[y == y_i].sum(0)
        return (p + 1) / ((y == y_i).sum() + 1)

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)
        self._r = sparse.csr_matrix(np.log(self.pr(x, 1, y) / self.pr(x, 0, y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self


def train_process():
    train = dp.train_file
    test = dp.test_file
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_label = dp.train_file
    subm = dp.submission_file
    train_features, _ = dp.process_data(train_file, test_file, submission_file )
    preds = np.zeros((len(test), len(label_cols)))
    for i, j in enumerate(label_cols):
        model = NbSvmClassifier(C=4, dual=True, n_jobs=-1).fit(train_features, train[j])
        r = model._r
        _, test_x = dp.build_matrix()
        preds[:, i] = model.predict_proba(test_x.multiply(r))[:, 1]

        model = NbSvmClassifier(C=4, dual=True, n_jobs=-1).fit(train_features, train_label)
    submid = pd.DataFrame({'id': subm["id"]})
    submission = pd.concat([submid, pd.DataFrame(preds, columns=label_cols)], axis=1)
    submission.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    train_process()