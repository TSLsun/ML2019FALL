#%%
import sys
import numpy as np
import pandas as pd
# np.set_printoptions(suppress=True)

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

# Use np.clip to prevent overflow
def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-6, 1-1e-6)

# Feature normalize, only on continues variable
def normalizeAll(x_train, x_test):
    x_all = np.concatenate((x_train, x_test), axis = 0)
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)
    np.save("mean106.npy", mean)
    np.save("std106.npy", std)

    index = [0, 1, 3, 4, 5]
    mean_vec = np.zeros(x_all.shape[1])
    std_vec = np.ones(x_all.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]
    x_all_nor = (x_all - mean_vec) / std_vec

    x_train_nor = x_all_nor[0:x_train.shape[0]]
    x_test_nor = x_all_nor[x_train.shape[0]:]

    return x_train_nor, x_test_nor

#%%
def normalize(X, sel_cols=[0,1,3,4,5], own_norm=False):
    if own_norm:
        mean = np.mean(X, axis = 0)
        std = np.std(X, axis = 0) 
    else:
        mean = np.load("mean106.npy")
        std = np.load("std106.npy")
    mean_vec = np.zeros(X.shape[1])
    std_vec = np.ones(X.shape[1])
    mean_vec[sel_cols] = mean[sel_cols]
    std_vec[sel_cols] = std[sel_cols]

    X_norm = (X - mean_vec) / std_vec

    return X_norm


#%%
if __name__ == '__main__':
    
    # normalizeAll(X, X_test)
    w = np.load("adagrad_w.npy")
    b = np.load("adagrad_b.npy")

    X_test = np.genfromtxt(X_test_file, delimiter=',', skip_header=1)
    x_test = normalize(X_test)
    y_pred = np.round(sigmoid(np.dot(x_test, w) + b))

    with open(output_csv, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_pred):
            f.write('%d,%d\n' %(i+1, v))
