#%%
import sys
import csv
import math
import pickle
import numpy as np
import pandas as pd

#%%
test_csv = sys.argv[1]
output_csv = sys.argv[2]
# test_csv = "data/testing_data.csv"
# output_csv = "ans.csv"

#%%
test_df = pd.read_csv(test_csv)
allFeatureName = test_df["測項"][:18].values
# print(allFeatureName, flush=True)
#%%
# featureMean = np.load("featureMean.npy")
with open('featureMean.pickle', 'rb') as handle:
    featureMean = pickle.load(handle)

mean = np.load("train_mean.npy")
std = np.load("train_std.npy")
W = np.load("weight.npy")


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

test_df_T = pd.DataFrame(data=tranverse_testdf(test_df, HR=9), columns=allFeatureName)

#%%
X_test = []
for i in range(0, test_df_T.shape[0], 9):
    x_features = test_df_T[allFeatureName].values[i:i+9].flatten()
    X_test.append(x_features)
X_test = np.array(X_test, dtype=np.float)

#%%
for i in range(X_test.shape[0]):
    for j in range(X_test.shape[1]):
        if std[j] != 0: 
            X_test[i][j] = (X_test[i][j]- mean[j]) / std[j]

#%%
X_test = np.concatenate((np.ones((len(X_test), 1)), X_test) , axis = 1)
#%%
### pred 
y_pred = np.dot(X_test, W)

#%%
# with open("origin_pred.csv", 'w', newline='') as csvfile:
    # w = csv.writer(csvfile)
    # w.writerow(['id','value']) 
    # for i in range(500):
        # content = ['id_'+str(i), y_pred[i][0]]
        # w.writerow(content)  

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
