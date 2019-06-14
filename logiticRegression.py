from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.model_selection import train_test_split
from config import SVDSIZE
from sklearn.metrics import accuracy_score, roc_auc_score
from gridsearch import grid_search
import numpy as np

data = np.load("./data/tfidf_svd_%s.npz" %(SVDSIZE, ))
x = data["x"]
y = data["y"]
y = y.reshape((-1, ))
params = [
    {
        "C": [ 0.001, 1E-4, 1E-5],
    }
]
for param in grid_search(params):
    print(param)
    lr = LogisticRegression(**param)
    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2)
    print("Start fitting...")
    lr.fit(train_x, train_y)
    print("Fitting conclude.")

    pred_prob_2 = lr.predict_proba(val_x)
    pred_prob = pred_prob_2[:, 1]
    pred_y = pred_prob > 0.5
    print("binary accuracy: ", accuracy_score(val_y, pred_y))
    print("roc_auc: ", roc_auc_score(val_y, pred_prob))