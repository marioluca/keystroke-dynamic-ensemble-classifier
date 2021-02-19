# Import required packages
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.callbacks import Callback
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
#import seaborn as sns
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import concatenate
from keras.layers import Concatenate,Flatten
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.layers import merge

n_classes = 100
ksplit = 10
n_epoch = 100
batchSize = 128
n_timesteps = 1
numpy_loss_history = np.zeros(n_epoch)
numpy_acc_history = np.zeros(n_epoch)
numpy_val_loss_history = np.zeros(n_epoch)
numpy_val_acc_history = np.zeros(n_epoch)

def createModelDNN(dim):

    inputs = Input(name="DNN-IN",batch_shape=(batchSize,n_features) )
    bn1 = BatchNormalization(name="DNN-BN1")(inputs)
    d11 = Dense(512, activation="relu",name="DNN-D1")(bn1)
    dol1 = Dropout(name="DNN-DO1",rate=0.2, seed=100)(d11)

    bn2 = BatchNormalization(name="DNN-BN2")(dol1)
    d12 = Dense(256, activation="relu", name="DNN-D2")(bn2)
    dol2 = Dropout(name="DNN-DO2", rate=0.2, seed=100)(d12)

    bn3 = BatchNormalization(name="DNN-BN3")(dol2)
    d13 = Dense(128, activation="relu", name="DNN-D3")(bn3)
    dol3 = Dropout(name="DNN-DO3", rate=0.2, seed=100)(d13)

    bn4 = BatchNormalization(name="DNN-BN4")(dol3)
    d14 = Dense(64, activation="relu", name="DNN-D4")(bn4)
    dol4 = Dropout(name="DNN-DO4", rate=0.2, seed=100)(d14)

    bn5 = BatchNormalization(name="DNN-BN5")(dol4)
    d15 = Dense(32, activation="relu", name="DNN-D5")(bn5)
    dol5 = Dropout(name="DNN-DO5", rate=0.2, seed=100)(d15)

    return Model(inputs,dol5)

def createModelLSTM(n_timestamps, n_features):
    inputShape = (batchSize, n_timestamps,n_features)
    inputs = Input(name="LSTM-IN",batch_shape=inputShape)
    x = BatchNormalization(name="LSTM-BN")(inputs)
    x = Bidirectional(LSTM(100, dropout=0.2,name="LSTM-L1"),name="LSTM-BI")(x)

    x = Dense(128,name="LSTM-D1")(x)
    x = Dense(64, name="LSTM-D2")(x)
    x = Dense(32, name="LSTM-D3")(x)
    x = Activation("relu",name="LSTM-A1")(x)
    model = Model(inputs, x)
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
# print("Unique values for outcome:", df["att65"].unique())
print("\nPercentage of distribution for outcome-")
print(df["USER_ID"].value_counts() / df.shape[0])
# print(df["att65"].value_counts()/df.shape[0])


# split the final dataset into train and test with 80:20
# x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:14].values, df['USER_ID'].values, test_size=0.2, random_state=2018)

x = df.iloc[:, 0:15].values
y = df.iloc[:, 15].values

min_max_scaler = MinMaxScaler(feature_range=(-1,1))
x = min_max_scaler.fit_transform(x)

# x_train, x_test, y_train, y_test = train_test_split(df.values, df['att65'].values, test_size=0.2, random_state=2018)
cv = StratifiedKFold(n_splits=ksplit)
# split the train dataset further into train and validation with 90:10
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, shuffle=True)

for train_index, test_index in cv.split(x, y):

    x_train = x[train_index]
    y_train = y[train_index]

    x_test = x[test_index]
    y_test = y[test_index]

    Y_train_labels, yTrainVal = pd.factorize(y_train)
    y_train = to_categorical(Y_train_labels, n_classes)

    Y_test_labels, yTestVal = pd.factorize(y_test)
    y_test = to_categorical(Y_test_labels, n_classes)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

    #create windowed datsets
    trainXwindowed = x_train.reshape(x_train.shape[0], n_timesteps, x_train.shape[1])
    trainYwindowed = y_train
    valXwindowed = x_val.reshape(x_val.shape[0], n_timesteps, x_val.shape[1])
    valYwindowed = y_val
    testXwindowed = x_test.reshape(x_test.shape[0], n_timesteps, x_test.shape[1])
    testYwindowed = y_test

    n_timestamps, n_features, n_outputs = trainXwindowed.shape[1], trainXwindowed.shape[2], trainYwindowed.shape[1]

    # create the LSTM and DNN output layers
    dnn = createModelDNN(x_train.shape[0])
    lstm = createModelLSTM(n_timestamps, n_features)

    print("dnn output: ", dnn.output, "Shape dnn: ", dnn.output.shape)
    print("lstm output: ", lstm.output, "Shape lstm: ", lstm.output.shape)

    # create the input to our final set of layers as the *output* of both
    # the LSTM and DNN

    combinedInput = Concatenate(axis=1)([dnn.output, lstm.output])

    print("combined input: ", combinedInput, "Shape cnn: ", combinedInput.shape)

    out = Dense(16, activation="relu")(combinedInput)
    out = Dense(n_classes, activation="softmax")(out)

    model = Model(inputs=[dnn.input, lstm.input], outputs=out)

    sgd = SGD(lr=0.1, momentum=0.09, decay=1e-6, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    #print(model.summary())
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    x_train = x_train[0:(x_train.shape[0] - x_train.shape[0] % batchSize), :]
    trainXwindowed = trainXwindowed[0:(trainXwindowed.shape[0] - trainXwindowed.shape[0] % batchSize), : , :]
    y_train = y_train[0:(y_train.shape[0] - y_train.shape[0] % batchSize), :]

    x_val = x_val[0:(x_val.shape[0] - x_val.shape[0] % batchSize), :]
    valXwindowed = valXwindowed[0:(valXwindowed.shape[0] - valXwindowed.shape[0] % batchSize), : , :]
    y_val = y_val[0:(y_val.shape[0] - y_val.shape[0] % batchSize), :]


    history_callback = model.fit([x_train, trainXwindowed], y_train, callbacks=[LossHistory()], verbose=0,
                                 batch_size=batchSize, epochs=n_epoch, validation_data=([x_val, valXwindowed], y_val))

    loss_history = history_callback.history["loss"]
    acc_history = history_callback.history["acc"]
    val_loss_history = history_callback.history["val_loss"]
    val_acc_history = history_callback.history["val_acc"]

    print("History: acc={0} val_acc={1} loss={2} val_loss={3}".format(acc_history,val_acc_history,loss_history,val_acc_history))


    current_numpy_loss_history = np.array(loss_history)
    numpy_loss_history = np.add(current_numpy_loss_history, numpy_loss_history)
    np.savetxt("/results/loss_history.txt", numpy_loss_history, delimiter=",")

    current_numpy_acc_history = np.array(acc_history)
    numpy_acc_history = np.add(current_numpy_acc_history, numpy_acc_history)
    np.savetxt("results/acc_history.txt", numpy_acc_history, delimiter=",")

    current_numpy_val_loss_history = np.array(val_loss_history)
    numpy_val_loss_history = np.add(current_numpy_val_loss_history, numpy_val_loss_history)
    np.savetxt("results/val_loss_history.txt", numpy_val_loss_history, delimiter=",")

    current_numpy_val_acc_history = np.array(val_acc_history)
    numpy_val_acc_history = np.add(current_numpy_val_acc_history, numpy_val_acc_history)
    np.savetxt("results/val_acc_history.txt", numpy_val_acc_history, delimiter=",")
    print("End of one split of cross validation")


numpy_loss_history = numpy_loss_history / ksplit
numpy_acc_history = numpy_acc_history / ksplit
numpy_val_loss_history = numpy_val_loss_history / ksplit
numpy_val_acc_history = numpy_val_acc_history / ksplit
total = [numpy_loss_history, numpy_acc_history, numpy_val_loss_history, numpy_val_acc_history]

np.savetxt("./results/all_statistics.csv", total, delimiter=", ")

#testing
#model.evaluate(x_test, y_test, batch_size=128, verbose=1)
