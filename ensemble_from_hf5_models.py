# load nets and make detection using a voting ensemble

from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import load_model
from matplotlib import pyplot
from numpy import mean
from numpy import std
import numpy
from numpy import array
from numpy import argmax

# load models from file
def read_hf5_classifiers(s, e, path='./models'):
    hf5s = list()
    for epoch in range(s, e):
        filename = path+'/model_' + str(epoch) + '.h5'
        hf5model = load_model(filename)
        hf5s.append(hf5model)
        print('>added %s' % filename)
    return hf5s

# perform multinomial classification using ensemble (majority)
def classify_ensemble(mf5_models, test_x):
    ypred = [model.predict(test_x) for model in mf5_models]
    ypred = array(ypred)
    summed = numpy.sum(ypred, axis=0)
    result = argmax(summed, axis=1)
    return result

# evaluate a specific number of models in the ensemble
def classify_n_models_in_ensemble(members, n_members, testX, testy):
    subset = members[:n_members]
    yhat = classify_ensemble(subset, testX)
    return accuracy_score(testy, yhat)

X, y = [], []  # change with dataset load from csv or pickle

# split data
n_train = 100
train_x, test_x = X[:n_train, :], X[n_train:, :]
train_y, test_y = y[:n_train], y[n_train:]
print(train_x.shape, test_x.shape)

# load hf5 sequence
hf5models = read_hf5_classifiers(1, 100)
print('added %d models' % len(hf5models))
hf5models = list(reversed(hf5models))

# evaluate different numbers of ensembles on hold out set
single_accs, ensemble_accs = [] , []
for i in range(1, len(hf5models) + 1):
    ensemble_res = classify_n_models_in_ensemble(hf5models, i, train_x, train_y)
    # classify with the i-th model
    test_y_encoded = to_categorical(train_y)
    _, model_res = hf5models[i - 1].evaluate(train_x, test_y_encoded, verbose=0)
    # aggregate data
    print('> %d: single=%.3f, ensemble=%.3f' % (i, model_res, ensemble_res))
    ensemble_accs.append(ensemble_res)
    single_accs.append(model_res)

# evaluate average/std acc of single models
print('Accuracy %.3f (%.3f)' % (mean(single_accs), std(ensemble_accs)))

# show results with respect to the number of models used for ensemble
x = [i for i in range(1, len(ensemble_accs) + 1)]
pyplot.plot(x, ensemble_accs, marker='o')
pyplot.plot(x, single_accs, marker='o', linestyle='None')
pyplot.show()
