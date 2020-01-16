import os
import numpy as np
import pandas as pd

fold = 'pred/'
csvs = [i for i in os.listdir(fold) if '_10.csv' in i]
df = pd.read_csv(fold + csvs[0]).values[:, 1:]
res = np.zeros((df.shape))

for i in csvs:
    df = pd.read_csv(fold+i)
    res += df.values[:, 1:]
res = res / len(csvs)
ans = np.argmax(res, axis=1)
df = pd.DataFrame({'id': np.arange(0, len(ans)), 'label': ans.astype('int')})
df.to_csv(fold + 'ensemble.csv', index=False)
