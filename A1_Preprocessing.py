import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

# Basic variables
N: int = 943
M: int = 1682
file = 'u.data'
col_names = ['user id', 'movie id', 'rating', 'timestamp']

# Import file to dataframe
df = pd.read_table(file, names=col_names, usecols=col_names[0:3], dtype=np.int32)
user_ratings = np.zeros([N, M])

# Process ratings and save to file
for i in range(0, N):   # foreach user in dataset
    # foreach rating of a unique user, centre and regularise data
    u = df.loc[df['user id'] == (i+1)]
    temp = np.array([k for j, k in zip(u['movie id'], u['rating'])])
    temp = temp.reshape(-1, 1)
    temp = StandardScaler().fit_transform(X=temp)
    # store (existing) ratings in array row, filling empty cells with 0
    x = 0
    for j, k in zip(u['movie id'], u['rating']):
        user_ratings[i, (j - 1)] = temp[x]
        x += 1
np.save('user_ratings.npy', user_ratings)
df.drop(columns=col_names[0:3])

# Process user ids
user_ids = np.zeros([N, N], dtype=np.int32)
for i in range(0, N):   # foreach user in dataset
    user_ids[i, i] = 1  # add one-hot encoded entry to input data array
np.save('user_ids.npy', user_ids)

# Split dataset into train, validation and test sets and save results to file
X_train, X_test, y_train, y_test = train_test_split(user_ids, user_ratings, test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.14)
np.savez('X_y_data.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# Split train data into 5 folds and save results to file
train_ind = []
test_ind = []
kfold = KFold(n_splits=5)
folds = kfold.split(X_train, y_train)
for j, (train, test) in enumerate(folds):
    train_ind.append(train)
    test_ind.append(test)
np.savez('folds.npz', train_ind=train_ind, test_ind=test_ind)
