import pandas as pd
import numpy as np
# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Basic variables
N: int = 943
M: int = 1682

# Load preprocessed data
user_ratings = np.load('user_ratings.npy')
user_ids = np.load('user_ids.npy')

data = np.load('X_y_data.npz', allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
data.close()

data = np.load('folds.npz', allow_pickle=True)
train_ind = data['train_ind']
test_ind = data['test_ind']
data.close()

# Train model for different values of H
print("hello")      # TODO remove
h = [3, 5, 100]
exp_metrics = np.zeros([3, 3])
i = 0
for H in h:
    RMSEs = []
    MAEs = []
    for j in range(0, 5):
        # Model
        model = Sequential()
        model.add(Dense(N, activation='tanh', input_dim=N))
        model.add(Dense(H, activation='tanh'))
        model.add(Dense(M, activation='tanh'))

        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))

        model.compile(loss='mean_squared_error', optimizer='sgd', metrics=[rmse, 'mae'])

        # Training
        history = model.fit(X_train[train_ind[j]], y_train[train_ind[j]], epochs=200, batch_size=50, verbose=0)

        print(j)        # TODO remove
        # Evaluation
        loss, rmse, mae = model.evaluate(X_train[test_ind[j]], y_train[test_ind[j]], verbose=0)

        # Store metric values
        RMSEs.append(rmse)
        MAEs.append(mae)

        # Plot metrics values at successive epochs
        if (i == 0) and (j == 0):
            plt.plot(history.history['rmse'])
            plt.plot(history.history['mae'])
            plt.title('Model Metrics')
            plt.ylabel('Metric')
            plt.xlabel('Epoch')
            plt.legend(['RMSE', 'MAE'], loc='upper left')
            plt.show()

        # Clear session, else may observe impacts of model resets (required by k-fold) on performance
        K.clear_session()
    # Store results of cycle in dataframe
    exp_metrics[i, 0] = H
    exp_metrics[i, 1] = np.average(RMSEs)
    exp_metrics[i, 2] = np.average(MAEs)
    i += 1

# Print results
table_cols = ['Hidden Layer Neurons (H)', 'RMSE', 'MAE']
df = pd.DataFrame(data=exp_metrics, columns=table_cols)
df = df.astype({'Hidden Layer Neurons (H)': np.int32})
print(df.to_string(formatters=({'Hidden Layer Neurons (H)': '{:d}'.format})))
# , 'RMSE': '{:,.5f}'.format, 'MAE': '{:,.5f}'.format})))
