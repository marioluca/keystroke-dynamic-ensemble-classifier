# Import required packages
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

n_classes = 100
batchSize = 128
epochsCount = 100

def createModel():
    model = Sequential()
    model.add(BatchNormalization(batch_input_shape=(x_train.shape[0],)))
    model.add(LSTM(256, dropout=0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    return model

# Dataset file
df = pd.read_csv('dataset-keystroke.csv')

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_epoch_end(self, epoch, logs={}):
        print("Epoch:{2} Loss:{0} Acc:{1} ValAcc:{3}".format(logs.get('loss'), logs.get('acc'), epoch, logs.get('val_acc')))

headers = df.head()

print('headers:', headers)
print("Unique values for outcome:", df["USER_ID"].unique())
print("\nPercentage of distribution for outcome-")
print(df["USER_ID"].value_counts() / df.shape[0])

x = df.iloc[:, 0:15].values
y = df.iloc[:, 15].values

min_max_scaler = MinMaxScaler(feature_range=(-1,1))
x = min_max_scaler.fit_transform(x)

cv = StratifiedKFold(n_splits=10)

for train_index, test_index in cv.split(x, y):

    x_train = x[train_index]
    y_train = y[train_index]

    x_test = x[test_index]
    y_test = y[test_index]

    Y_train_labels, yTrainVal = pd.factorize(y_train)
    y_train = to_categorical(Y_train_labels, n_classes)

    Y_train_labels, yTrainVal = pd.factorize(y_test)
    y_test = to_categorical(Y_train_labels, n_classes)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

    trainXwindowed = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
    trainYwindowed = y_train
    valXwindowed = x_val.reshape(x_val.shape[0],1,x_val.shape[1])
    valYwindowed = y_val
    testXwindowed = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    testYwindowed = y_test

    n_timestamps, n_features, n_outputs = trainXwindowed.shape[1], trainXwindowed.shape[2], trainYwindowed.shape[1]
    model = Sequential()
    model.add(BatchNormalization(batch_input_shape=(batchSize,n_timestamps, n_features)))
    model.add(LSTM(128, dropout=0.2,return_sequences=True))
    model.add(LSTM(64, dropout=0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation='softmax'))

    sgd = SGD(lr=0.1, momentum=0.09, decay=1e-6, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    # Train the model
    history_callback = model.fit(trainXwindowed, trainYwindowed, callbacks=[LossHistory()],
                                 verbose=1, batch_size=batchSize, epochs=epochsCount, validation_data=(valXwindowed, valYwindowed))

    loss_history = history_callback.history["loss"]
    acc_history = history_callback.history["acc"]
    val_loss_history = history_callback.history["val_loss"]
    val_acc_history = history_callback.history["val_acc"]

    print("History: acc={0} val_acc={1} loss={2} val_loss={3}".format(acc_history,val_acc_history,loss_history,val_acc_history))

    model.evaluate(x_test, y_test, batch_size=batchSize, verbose=1)
