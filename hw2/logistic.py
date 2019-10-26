#%%
import sys
import numpy as np
import pandas as pd
# np.set_printoptions(suppress=True)

### comment when submit
# import matplotlib.pyplot as plt
###

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
print("X.shape", X.shape)
print("Y.shape", Y.shape)
print("X_test.shape", X_test.shape)
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
# Gradient descent using adagrad
def train(X, Y, lr=0.05, epoch=100000, valid_size=0.2, printStep=100, early_stop_steps=50):
    x_train, x_valid, y_train, y_valid = train_valid_split(X, Y, valid_size=valid_size, random_state=42)
    print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

    b = 0.0
    w = np.zeros(x_train.shape[1])
    # lr = 0.05
    # epoch = 100 
    b_lr = 0
    w_lr = np.ones(x_train.shape[1])
    
    min_val_cost = 1e10
    cnt = 0
    history = {
        'loss': [],
        'val_loss': [],
    }
    for e in range(epoch):
        z = np.dot(x_train, w) + b
        pred = sigmoid(z)
        loss = y_train - pred

        b_grad = -1*np.sum(loss)
        w_grad = -1*np.dot(loss, x_train)

        b_lr += b_grad**2
        w_lr += w_grad**2


        b = b-lr/np.sqrt(b_lr)*b_grad
        w = w-lr/np.sqrt(w_lr)*w_grad

        loss = -1*np.mean(y_train*np.log(pred+1e-100) + (1-y_train)*np.log(1-pred+1e-100))
        valid_pred = sigmoid(np.dot(x_valid, w) + b)
        val_loss = -1*np.mean(y_valid*np.log(valid_pred+1e-100) + (1-y_valid)*np.log(1-valid_pred+1e-100))
        history['loss'].append(loss)
        history['val_loss'].append(val_loss)

        if(e+1)%printStep == 0:
            print('epoch:{}\nloss:{} | val_loss:{}\n'.format(e+1, loss, val_loss))

        if val_loss < min_val_cost:
            min_val_cost = val_loss
            cnt = 0
        else:
            cnt += 1

        if cnt > early_stop_steps:
            print(f"early stop! val_loss {min_val_cost} don't decrease in {early_stop_steps} iters")
            break 

    return w, b, history

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
### comment when submit
# def plot_history(h):
    # plt.figure()
    # epochs = range(len(h['loss']))
    # plt.plot(epochs, h['loss'], label='train')
    # plt.plot(epochs, h['val_loss'], label='valid')
    # plt.legend()
    # plt.show()
###
#%%
if __name__ == '__main__':
    
    # normalizeAll(X, X_test)
    x = normalize(X)
    w, b, history = train(x, Y, early_stop_steps=20)
    
    # plot_history(history)
    y_train_pred = np.round(sigmoid(np.dot(x, w) + b)) 
    print(accuracy(Y, y_train_pred))

    x_test = normalize(X_test)
    y_pred = np.round(sigmoid(np.dot(x_test, w) + b))

    with open(output_csv, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_pred):
            f.write('%d,%d\n' %(i+1, v))
#%%
