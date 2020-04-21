import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# Basic variables
N: int = 943
M: int = 1682
col_names = ['user id', 'movie id', 'rating', 'timestamp']

# Import file to dataframe
print('Importing and processing data.')
df = pd.read_table('u.data', names=col_names, usecols=col_names[0:3], dtype=np.int32)

# Process ratings and save to file
user_ratings = np.zeros([N, M])
for i in range(0, N):   # foreach user in dataset
    # foreach rating of a unique user, centre and normalise data
    u = df.loc[df['user id'] == (i+1)]
    temp = np.array([k for j, k in zip(u['movie id'], u['rating'])])
    temp = temp.reshape(-1, 1)
    temp = StandardScaler(with_std=False).fit_transform(X=temp)
    temp = temp.reshape(len(temp))
    min_r = temp.min()
    max_r = temp.max()
    x = 0
    for j, k in zip(u['movie id'], u['rating']):
        # store (existing) ratings in array row, filling empty cells with 0
        user_ratings[i, (j - 1)] = np.interp(temp[x], [min_r, max_r], [0, 1])
        x += 1
np.save('user_ratings.npy', user_ratings)
df.drop(columns=col_names[0:3])

# Process user ids
# one-hot encoded set of ids equals NxN identity matrix
user_ids = np.identity(N, dtype=np.int32)
np.save('user_ids.npy', user_ids)

# Split train data into 5 folds and save results to file
train_ind = []
test_ind = []
kfold = KFold(n_splits=5, shuffle=True)
folds = kfold.split(user_ids, user_ratings)
for j, (train, test) in enumerate(folds):
    train_ind.append(train)
    test_ind.append(test)
np.save('train_ind.npy', train_ind)
np.save('test_ind.npy', test_ind)
print('Done.')
