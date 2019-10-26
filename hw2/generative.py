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

#%%
X = np.genfromtxt(X_train_file, delimiter=',', skip_header=1)
Y = np.genfromtxt(Y_train_file, delimiter=',', skip_header=0)

X_test = np.genfromtxt(X_test_file, delimiter=',', skip_header=1)
dim = 106
# print("X.shape", X.shape)
# print("Y.shape", Y.shape)
# print("X_test.shape", X_test.shape)
#%%
def train_valid_split(X, y, valid_size=0.2, random_state=42):
    n_train = int((1 - valid_size) * len(X))
    n_test = len(X) - n_train
    n_samples = len(X)

    rng = np.random.RandomState(random_state)
    permutation = rng.permutation(n_samples)
    ind_test = permutation[:n_test]
    ind_train = permutation[n_test:(n_test + n_train)]

    X_train = X[ind_train]
    X_valid = X[ind_test]
    y_train = y[ind_train]
    y_valid = y[ind_test]

    return X_train, X_valid, y_train, y_valid
# X_train, X_valid, y_train, y_valid = train_valid_split(X, Y, valid_size=0.2, random_state=42)
# print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

#%%
# Use np.clip to prevent overflow
def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-6, 1-1e-6)

# Feature normalize, only on continues variable
def normalizeAll(x_train, x_test):
    x_all = np.concatenate((x_train, x_test), axis = 0)
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)
    # np.save("mean106.npy", mean)
    # np.save("std106.npy", std)

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
def train(x_train, y_train):
    cnt1 = 0
    cnt2 = 0
    
    mu1 = np.zeros((dim,))
    mu2 = np.zeros((dim,))
    
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            cnt1 += 1
            mu1 += x_train[i]
        else:
            cnt2 += 1
            mu2 += x_train[i]
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((dim,dim))
    sigma2 = np.zeros((dim,dim))
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            sigma1 += np.dot(np.transpose([x_train[i] - mu1]), [(x_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([x_train[i] - mu2]), [(x_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2

    
    share_sigma = (cnt1 / x_train.shape[0]) * sigma1 + (cnt2 / x_train.shape[0]) * sigma2
    return mu1, mu2, share_sigma, cnt1, cnt2
def predict(x_test, mu1, mu2, share_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(share_sigma)

    w = np.dot( (mu1-mu2), sigma_inverse)
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inverse), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inverse), mu2) + np.log(float(N1)/N2)

    z = np.dot(w, x_test.T) + b
    pred = sigmoid(z)
    return pred
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

def accuracy(Y_pred, Y_label):
    acc = np.sum(Y_pred == Y_label)/len(Y_pred)
    return acc

#%%
if __name__ == '__main__':
    
    x, x_test = normalizeAll(X, X_test)
    # x = normalize(X)    
    mu1, mu2, shared_sigma, N1, N2 = train(x, Y)
    
    # y_pred = predict(x, mu1, mu2, shared_sigma, N1, N2)
    # y_pred = np.around(y_pred)
    # result = (Y == y_pred)
    # print('Train acc = %f' % (float(result.sum()) / result.shape[0])) 
    
    x_test = normalize(X_test)
    y_test_pred = predict(x_test, mu1, mu2, shared_sigma, N1, N2)
    y_test_pred = np.around(y_test_pred)
#%%
    with open(output_csv, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_test_pred):
            f.write('%d,%d\n' %(i+1, v))

#%%
