import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
from tensorflow.keras.activations import relu
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Basic variables
N: int = 943
M: int = 1682

# Load preprocessed data
user_ratings = np.load('user_ratings.npy')
user_ids = np.load('user_ids.npy')
train_ind = np.load('train_ind.npy', allow_pickle=True)
test_ind = np.load('test_ind.npy', allow_pickle=True)

H = 15
lr = 0.1
m = 0.6
weight_decay = [0.1, 0.5, 0.9]
exp_metrics = np.zeros([len(weight_decay), 3])
i = 0
for r in weight_decay:
    RMSEs = []
    MAEs = []
    print(r)  # TODO remove
    for j in range(0, 5):
        # Model
        model = Sequential()
        # Input layer
        model.add(Dense(N, kernel_initializer=initializers.Ones(),
                        bias_initializer=initializers.Zeros(), input_dim=N))
        # Leaky ReLU activation function
        LRelU = lambda x: relu(x, alpha=0.01)
        # Hidden layer
        model.add(Dense(H, kernel_initializer=initializers.glorot_uniform(),
                        bias_initializer=initializers.glorot_uniform(), activation=LRelU,
                        kernel_regularizer=l1(r), bias_regularizer=l1(r)))
        # Output layer
        model.add(Dense(M, kernel_initializer=initializers.glorot_uniform(),
                        bias_initializer=initializers.glorot_uniform(), activation='sigmoid'))

        # SGD optimiser
        sgd = SGD(learning_rate=lr, momentum=m, nesterov=False)

        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))

        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=[rmse, 'mae'])

        # Termination Criterion
        term = callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10)

        # Training
        history = model.fit(user_ids[train_ind[j]], user_ratings[train_ind[j]],
                            epochs=150, batch_size=10, callbacks=[term], verbose=0)

        # Evaluation
        loss, rmse, mae = model.evaluate(user_ids[test_ind[j]], user_ratings[test_ind[j]], verbose=0)

        # Store metric values
        RMSEs.append(rmse)
        MAEs.append(mae)

        # Plot metrics values at end of train-test cycle
        fig = plt.figure(dpi=300, edgecolor='black')
        plt.plot(history.history['rmse'])
        plt.plot(history.history['mae'])
        plt.plot(history.history['loss'])
        plt.title('Model Metrics | r={0} | CV Iteration #{1}'.format(r, j + 1))
        plt.ylabel('Metric')
        plt.xlabel('Epoch')
        plt.legend(['RMSE', 'MAE', 'MSE'], loc='upper right')
        plt.show()
        # Save to file in png format (report)
        fig.savefig(fname=f'./a4_plots/a4_r_{i + 1}_cv_{j + 1}')

        # Clear session, else may observe impacts of model resets (required by k-fold) on performance
        K.clear_session()
    # Store results of cycle in dataframe
    exp_metrics[i, 0] = r
    exp_metrics[i, 1] = np.mean(RMSEs)
    exp_metrics[i, 2] = np.mean(MAEs)
    i += 1

# Print results
table_cols = ['r', 'RMSE', 'MAE']
df = pd.DataFrame(data=exp_metrics, columns=table_cols)
print(df.to_string(index=False, formatters=({'RMSE': '{:,.5f}'.format, 'MAE': '{:,.5f}'.format})))
