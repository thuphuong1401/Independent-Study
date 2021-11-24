import numpy as np
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

from src.loader import CumuloDataset
from src.utils import make_directory


data_dir = '/scratch/pvu3/Indep Study/netcdf/cumulo-tiles/npz/label/'
dataset = CumuloDataset(root_dir=data_dir, ext="npz")

xs = []
ys = []

for d in dataset:
    xs.append(d[1]) # radiances
    ys.append(d[4]) # labels

shape = xs[0].shape
xs = np.vstack(xs).reshape(-1, shape[1] * shape[2] * shape[3]) # flatten tiles
ys = np.hstack(ys)

print(xs.shape, ys.shape)

print(min(ys), max(ys))

from sklearn.model_selection import train_test_split

train_xs, test_xs, train_ys, test_ys = train_test_split(xs, ys, test_size=0.20, random_state=42)

train_xs, val_xs, train_ys, val_ys = train_test_split(train_xs, train_ys, test_size=0.125, random_state=42)

print(train_xs.shape, train_ys.shape, val_xs.shape, val_ys.shape, test_xs.shape, test_ys.shape)

import lightgbm as lgb

lgb_train = lgb.Dataset(train_xs, train_ys)
lgb_valid = lgb.Dataset(val_xs, val_ys)

params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'verbose': -1,
    'num_classes': 8,
    "num_iterations": 20,
}

gbm = lgb.train(params, lgb_train, valid_sets=[lgb_valid])

train_prob_pred = gbm.predict(train_xs, num_iteration = gbm.best_iteration)
val_prob_pred = gbm.predict(val_xs, num_iteration = gbm.best_iteration)
test_prob_pred = gbm.predict(test_xs, num_iteration = gbm.best_iteration)

print("Shape {}".format(train_prob_pred.shape))

# take argmax of prediction probabilities
train_y_pred = np.argmax(train_prob_pred, 1)
print("Shape {}".format(train_y_pred.shape))
val_y_pred = np.argmax(val_prob_pred, 1)
test_y_pred = np.argmax(test_prob_pred, 1)

accuracy = accuracy_score(test_ys, test_y_pred)
print("Accuracy is {}".format(accuracy))

from sklearn.metrics import confusion_matrix

train_cm = confusion_matrix(train_ys, train_y_pred, labels=range(8))
val_cm = confusion_matrix(val_ys, val_y_pred, labels=range(8))
test_cm = confusion_matrix(test_ys, test_y_pred, labels=range(8))

# np.save("train-confusion-matrix.npy", train_cm)
# np.save("test-confusion-matrix.npy", test_cm)
# np.save("val-confusion-matrix.npy", val_cm)

def normalize_confusion_matrix(darray):
    return (darray / np.sum(darray, 1, keepdims=True)).round(2)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

#%matplotlib inline

# best model, batch weights
train_cm, val_cm, test_cm = normalize_confusion_matrix(train_cm), normalize_confusion_matrix(val_cm), normalize_confusion_matrix(test_cm)

plt.figure(figsize = (20,5))

plt.subplot(131)

df_cm = pd.DataFrame(train_cm, index = range(8), columns = range(8))

plt.title("TRAIN")
ax = sn.heatmap(df_cm, annot=True, vmin=0, vmax=1)
ax.set(xlabel='predicted', ylabel='target')

plt.subplot(132)

df_cm = pd.DataFrame(val_cm, index = range(8), columns = range(8))

plt.title("VAL")
ax = sn.heatmap(df_cm, annot=True, vmin=0, vmax=1)
ax.set(xlabel='predicted', ylabel='target')

plt.subplot(133)

df_cm = pd.DataFrame(test_cm, index = range(8), columns = range(8))

plt.title("TEST")
ax = sn.heatmap(df_cm, annot=True, vmin=0, vmax=1)
ax.set(xlabel='predicted', ylabel='target')

plt.savefig("lightgbm-confusion-matrices.png", bboxes="tight")

make_directory("results/lgbm/")
gbm.save_model("results/lgbm/lightgbm-model.txt")
