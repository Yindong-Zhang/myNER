from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from gridsearch import grid_search
from config import SVDSIZE

data = np.load("./data/tfidf_svd_%s.npz" %(SVDSIZE, ))
x = data["x"]
y = data["y"]
y = y.reshape((-1, ))
params =[
    {
        "probability": [True, ],
        "C": [1, 0.1, 0.01, 0.001],
        "gamma": [1/0.1, 1/1.6, 1/3.2, 1/6.4]
    }
]
for param in grid_search(params):
    print(param)
    svc = SVC(**param)

    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size= 0.2)
    print("Start fitting...")
    svc.fit(train_x, train_y)
    print("Fitting conclude.")

    pred_prob_2 = svc.predict_proba(val_x)
    pred_prob = pred_prob_2[:, 1]
    pred_y = pred_prob> 0.5
    print("binary accuracy: ", accuracy_score(val_y, pred_y))
    print("roc_auc: ", roc_auc_score(val_y, pred_prob))
