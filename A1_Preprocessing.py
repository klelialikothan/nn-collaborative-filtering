import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

N: int = 943
M: int = 1682
file = 'u.data'
col_names = ['user id', 'movie id', 'rating', 'timestamp']
df = pd.read_table(file, names=col_names, usecols=col_names[0:3], dtype=np.int32)
df = df.sort_values(by=['user id', 'movie id'])
ratings_list = []
for i in range(0, N):
    temp = np.zeros(M)
    u = df.loc[df['user id'] == (i+1)]
    rated = np.array([k for j, k in zip(u['movie id'], u['rating'])])
    rated = rated.reshape(-1, 1)
    rated = StandardScaler().fit_transform(X=rated)
    x = 0
    for j, k in zip(u['movie id'], u['rating']):
        temp[j-1] = rated[x]
        x += 1
    ratings_list.append(temp)
df.drop(columns=col_names[0:3])
user_ids = []
for i in range(0, N):
    temp = np.zeros(N, dtype=np.int32)
    temp[i] = 1
    user_ids.append(temp)
X_train, X_test, y_train, y_test = train_test_split(user_ids, ratings_list, test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.14)
kfold = StratifiedKFold(n_splits=5)
