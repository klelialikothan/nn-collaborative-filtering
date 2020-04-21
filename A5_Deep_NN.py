import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import initializers
from tensorflow.keras.activations import relu
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

h1 = [20, 15, 15, 15]
h2 = [20, 15, 15, 10]
h3 = [20, 15, 10, 5]
lr = 0.1
m = 0.6
exp_metrics = np.zeros([len(h1), 5])
i = 0
for H1, H2, H3 in zip(h1, h2, h3):
    RMSEs = []
    MAEs = []
    print(f'Training model with H1={H1}, H2={H2}, H3={H3} ...')
    for j in range(0, 5):
        # Model
        model = Sequential()
        # Input layer
        model.add(Dense(N, kernel_initializer=initializers.Ones(),
                        bias_initializer=initializers.Zeros(), input_dim=N))
        # Leaky ReLU activation function
        def lrelu(x): return relu(x, alpha=0.01)
        # Hidden layer 1
        model.add(Dense(H1, kernel_initializer=initializers.glorot_uniform(),
                        bias_initializer=initializers.glorot_uniform(), activation=lrelu))
        model.add(Dropout(0.2))
        # Hidden layer 2
        model.add(Dense(H2, kernel_initializer=initializers.glorot_uniform(),
                        bias_initializer=initializers.glorot_uniform(), activation=lrelu))
        model.add(Dropout(0.3))
        # Hidden layer 3
        model.add(Dense(H3, kernel_initializer=initializers.glorot_uniform(),
                        bias_initializer=initializers.glorot_uniform(), activation=lrelu))
        model.add(Dropout(0.4))
        # Output layer
        model.add(Dense(M, kernel_initializer=initializers.glorot_uniform(),
                        bias_initializer=initializers.glorot_uniform(), activation='sigmoid'))

        # SGD optimiser
        sgd = SGD(learning_rate=lr, momentum=m, nesterov=False)

        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))

        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=[rmse, 'mae'])

        # Termination Criterion
        term = callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=20)

        # Training
        history = model.fit(user_ids[train_ind[j]], user_ratings[train_ind[j]],
                            epochs=200, batch_size=10, callbacks=[term], verbose=0)

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
        plt.title('Model Metrics | H1={0} | H2={1} | H3={2} | CV Iteration #{3}'.format(H1, H2, H3, j + 1))
        plt.ylabel('Metric')
        plt.xlabel('Epoch')
        plt.legend(['RMSE', 'MAE', 'MSE'], loc='upper right')
        plt.show()
        # Save to file in png format (report)
        # fig.savefig(fname=f'./a5_plots/a5_set_{i + 1}_cv_{j + 1}')

        # Clear session, else may observe impacts of model resets (required by k-fold) on performance
        K.clear_session()
    # Store results of cycle in dataframe
    exp_metrics[i, 0] = H1
    exp_metrics[i, 1] = H2
    exp_metrics[i, 2] = H3
    exp_metrics[i, 3] = np.mean(RMSEs)
    exp_metrics[i, 4] = np.mean(MAEs)
    i += 1

# Print results
print('Training done.\n')
table_cols = ['H1', 'H2', 'H3', 'RMSE', 'MAE']
df = pd.DataFrame(data=exp_metrics, columns=table_cols)
df = df.astype({'H1': np.int32, 'H2': np.int32, 'H3': np.int32})
print(df.to_string(index=False, formatters=({'RMSE': '{:,.5f}'.format, 'MAE': '{:,.5f}'.format})))
