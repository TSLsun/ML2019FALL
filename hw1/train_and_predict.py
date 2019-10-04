#%%
import numpy as np
import pandas as pd
import math
import csv
import sys
import pickle

#%%
train_data1_csv = sys.argv[1]
train_data2_csv = sys.argv[2]
test_csv = sys.argv[3]
output_csv = sys.argv[4]
# train_data1_csv = "data/year1-data.csv"
# train_data2_csv = "data/year2-data.csv" 
# test_csv = "data/testing_data.csv"
# output_csv = "output.csv"

#%%
df  = pd.read_csv(train_data1_csv)
df2 = pd.read_csv(train_data2_csv)
#%%
allFeatureName = df["測項"][:18].values
print(allFeatureName, flush=True)

#%%
def data_process(df_values, HR=24):
    """ Tranverse dataFrame into hour data 
    Args:
        df.values (np.ndarray): raw data

    Returns:
        data (np.ndarray): output data
    """
    data = np.zeros((int(len(df_values)/18 * 24), 18))
    cnt = 0
    for row in range(0, df_values.shape[0], 18):
        for hr in range(HR):
            for feature in range(18):
                # clean data 'NR' -> 0, replace '*#x' -> ''
                curStr = df_values[row+feature][hr+2]
                if curStr == 'NR':
                    curStr = 0
                else:
                    if type(curStr)==str:
                        ### remove str contain '*#x'
                        if '*' in curStr or '#' in curStr or 'x' in curStr:
                            curStr = np.nan
                data[cnt][feature] = float(curStr)
            cnt += 1
    return data

#%%
dataset  = pd.DataFrame(data=data_process(df.values), columns=allFeatureName)
dataset2 = pd.DataFrame(data=data_process(df2.values), columns=allFeatureName)

#%% 
mergeDataset = pd.concat([dataset, dataset2])
print(mergeDataset.shape, flush=True)

#%%
print(f"Feature           min      max      mean")
print("-"*40)
featureMean = {}
for feature in allFeatureName:
    print(f"{feature:12s}  {min(mergeDataset[feature]):7.2f}, {max(mergeDataset[feature]):7.2f}, {np.mean(mergeDataset[feature]):7.2f}")
    featureMean[feature] = np.mean(mergeDataset[feature]) 

#%%
with open('featureMean.pickle', 'wb') as handle:
    pickle.dump(featureMean, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
def parse2train(df, select_features=allFeatureName):
    X = []
    y = []
    # select_features = ["PM2.5"]
    for i in range(len(df) - 9):
        x_features = df[select_features].values[i:i+9].flatten()
        y_feature = df['PM2.5'].values[i+9]
        if np.isnan(x_features).any() or np.isnan(y_feature).any():
            continue  # data contain NaN
        X.append(x_features)
        y.append(y_feature)
    X = np.array(X, dtype=np.float)
    y = np.array(y, dtype=np.float).reshape(-1,1)

    return X, y

X, y = parse2train(df=mergeDataset, select_features=allFeatureName)
print(X.shape, y.shape, flush=True)
#%%
### normalize 
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if std[j] != 0: 
            X[i][j] = (X[i][j]- mean[j]) / std[j]

np.save("train_mean.npy", mean)
np.save("train_std.npy", std)
#%% 
## add bias to X
X = np.concatenate((np.ones((len(X), 1)), X) , axis = 1)
print(X.shape, y.shape, flush=True)

#%%
def train_test_split(X, y, test_size=0.2, random_state=42):
    n_train = int((1 - test_size) * len(X))
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
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

#%%
# Adam
# ref: https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L467
# ref: https://sefiks.com/2018/06/23/the-insiders-guide-to-adam-optimization-algorithm-for-deep-learning/
# ref: https://gluon.mxnet.io/chapter06_optimization/adam-scratch.html
def adam(X, y, X_valid=None, y_valid=None, lr=1e-3, num_iter=100, early_stop_steps=50, print_steps=10):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    mt = np.zeros((X.shape[1],1))
    vt = np.zeros((X.shape[1],1))
    w = np.zeros((X.shape[1],1))
    past_cost = {
                    'train_cost': [],
                    'valid_cost': [],        
                }
    min_val_cost = 1e10
    cnt = 0
    for t in range(1, num_iter+1):
        y_ = np.dot(X, w)
        loss = y_ - y
        grad = 2 * np.dot(X.transpose(), loss)
        
        # w -= lr * grad 
        mt = beta1 * mt + (1-beta1) * grad
        vt = beta1 * vt + (1-beta2) * (grad ** 2)

        mt_hat = mt / (1 - beta1 ** t)
        vt_hat = vt / (1 - beta2 ** t)
        
        w = w - lr * mt_hat / (np.sqrt(vt_hat) + eps)
        
        cost = np.dot(loss.transpose(), loss)[0][0] / len(X)
        cost_sqrt = np.sqrt(cost)
        
        if X_valid is not None and y_valid is not None:
            val_loss = np.dot(X_valid, w) - y_valid
            val_cost = np.dot(val_loss.transpose(), val_loss)[0][0] / len(X_valid) 
            val_cost_sqrt = np.sqrt(val_cost)
            if t % print_steps == 0: 
                print(f"Iteration {t} | train_loss: {cost_sqrt:.3f}, val_loss: {val_cost_sqrt:.3f}", flush=True)

            past_cost['train_cost'].append(cost_sqrt)
            past_cost['valid_cost'].append(val_cost_sqrt)

            if past_cost['valid_cost'][-1] < min_val_cost:
                min_val_cost = past_cost['valid_cost'][-1]
                cnt = 0
            else:
                cnt += 1

            if cnt > early_stop_steps:
                print(f"early stop! val_loss don't decrease in {early_stop_steps} iters")
                break 
        else:    
            if t % print_steps == 0: 
                print(f"Iteration {t} | train_loss: {cost_sqrt:.3f}", flush=True)
            past_cost['train_cost'].append(cost_sqrt)

    np.save("weight.npy", w)

    return w, past_cost
#%%
W, past_cost = adam(X_train, y_train, X_valid, y_valid, lr=0.003, num_iter=50000, early_stop_steps=10)


#%%
### TESTING PART
def tranverse_testdf(df, HR=9):
    """ Tranverse dataFrame into hour data 
    Args:
        df(pd.DataFrame): raw test data

    Returns:
        all_data (list): output data
    """
    all_data = []
    for row in range(0, df.shape[0], 18):
        for hr in range(HR):
            data = []
            for feature in range(18):
                # clean data 'NR' -> 0, replace '*#x' -> ''
                curStr = df.loc[row+feature][hr+2]
                if curStr == 'NR':
                    curStr = 0
                else:
                    if type(curStr)==str:
                        ### replace '*#x' with ''
                        # curStr = curStr.replace('*','').replace('x','').replace('#','')
                        
                        ### replace str contain '*#x' with mean
                        if '*' in curStr or '#' in curStr or 'x' in curStr:
                            curStr = featureMean[allFeatureName[feature]]
                    else:
                        if np.isnan(float(curStr)):
                            curStr = featureMean[allFeatureName[feature]]
                
                data.append(float(curStr))
            all_data.append(data)
    return all_data
#%%
# test_df = pd.read_csv("data/testing_data.csv")
test_df = pd.read_csv(test_csv)
test_df_T = pd.DataFrame(data=tranverse_testdf(test_df, HR=9), columns=allFeatureName)
test_df_T.head()

#%%
X_test = []
for i in range(0, test_df_T.shape[0], 9):
    x_features = test_df_T[allFeatureName].values[i:i+9].flatten()
    X_test.append(x_features)
X_test = np.array(X_test, dtype=np.float)
X_test.shape


#%%
for i in range(X_test.shape[0]):
    for j in range(X_test.shape[1]):
        if std[j] != 0: 
            X_test[i][j] = (X_test[i][j]- mean[j]) / std[j]

#%%
X_test = np.concatenate((np.ones((len(X_test), 1)), X_test) , axis = 1)
print(X_test.shape, flush=True)
#%%
y_pred = np.dot(X_test, W)

#%%
# with open('output.csv', 'w', newline='') as csvfile:
with open("origin_pred.csv", 'w', newline='') as csvfile:
    w = csv.writer(csvfile)
    w.writerow(['id','value']) 
    for i in range(500):
        content = ['id_'+str(i), y_pred[i][0]]
        w.writerow(content)  

#%%
y_pred_mean = np.mean(y_pred[np.isnan(y_pred)==False])

with open(output_csv, 'w', newline='') as csvfile:
    w = csv.writer(csvfile)
    w.writerow(['id','value']) 
    for i in range(500):
        if np.isnan(y_pred[i][0]) or y_pred[i][0] < 0:
            content = ['id_'+str(i), y_pred_mean]
        else:
            content = ['id_'+str(i), y_pred[i][0]]
        w.writerow(content)  

#%%
