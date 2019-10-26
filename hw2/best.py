#%%
import sys
import csv
import numpy as np
import pandas as pd
import itertools
# np.set_printoptions(suppress=True)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.externals import joblib

#%%
raw_data = sys.argv[1]
test_data = sys.argv[2]
X_train_file = sys.argv[3]
Y_train_file = sys.argv[4]
X_test_file = sys.argv[5]
output_csv = sys.argv[6]
# X_train_file = "data/X_train"
# Y_train_file = "data/Y_train"
# X_test_file = "data/X_test"
# output_csv = "adagrad_pred.csv"

#%%
# X = np.genfromtxt(X_train_file, delimiter=',', skip_header=1)
# Y = np.genfromtxt(Y_train_file, delimiter=',', skip_header=0)
X = pd.read_csv(X_train_file)
Y = np.genfromtxt(Y_train_file, delimiter=',', skip_header=0)

X_test = np.genfromtxt(X_test_file, delimiter=',', skip_header=1)
# print("X.shape", X.shape)
# print("Y.shape", Y.shape)
# print("X_test.shape", X_test.shape)

#%%
min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(X)

# model = GradientBoostingClassifier(criterion='friedman_mse', init=None, learning_rate=0.025,
	 # loss='deviance', max_depth=18, max_features='sqrt', max_leaf_nodes=None,
	 # min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=20,
	 # min_samples_split=200, min_weight_fraction_leaf=0.0, n_estimators=500,
	 # n_iter_no_change=None, presort='auto', random_state=42, subsample=0.8,
	 # tol=0.0001, validation_fraction=0.1, verbose=0, warm_start=True)
#%%
# model.fit(x_scaled, Y)

# joblib.dump(model, "best.joblib")
model = joblib.load("best.joblib")


x_test = min_max_scaler.fit_transform(X_test)
y_pred = model.predict(x_test)

with open(output_csv, 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(y_pred):
        f.write('%d,%d\n' %(i+1, v))
