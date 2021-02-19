
import pprint as pp
import pandas as pd
import numpy as np

import math

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.callbacks import Callback,EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering,SpectralClustering
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- Voter ---------------------------------------------
class VotingModel(object):

    def __init__(self, model_list, voting='hard',
                 weights=None, nb_classes=None):
        """(Weighted) majority vote model for a given list of Keras models.

        Parameters
        ----------
        model_list: An iterable of Keras models.
        voting: Choose 'hard' for straight-up majority vote of highest model probilities or 'soft'
            for a weighted majority vote. In the latter, a weight vector has to be specified.
        weights: Weight vector (numpy array) used for soft majority vote.
        nb_classes: Number of classes being predicted.

        Returns
        -------
        A voting model that has a predict method with the same signature of a single keras model.
        """
        self.model_list = model_list
        self.voting = voting
        self.weights = weights
        self.nb_classes = nb_classes

        if voting not in ['hard', 'soft']:
            raise Exception('Voting has to be either hard or soft')

        if weights is not None:
            if len(weights) != len(model_list):
                raise ('Number of models {0} and length of weight vector {1} has to match.'
                       .format(len(weights), len(model_list)))

    def predict(self, X, batch_size=128, verbose=0):
        predictions = list(map(lambda model: model.predict(X, batch_size, verbose), self.model_list))
        nb_preds = len(X)
        for d in range(0,len(predictions)):
            predictions[d] = predictions[d][:, 0:g_n_classes]
        if self.voting == 'hard':
            for i, pred in enumerate(predictions):
                pred = list(map(
                    lambda probas: np.argmax(probas, axis=-1), pred
                ))
                predictions[i] = np.asarray(pred).reshape(nb_preds, 1)
            argmax_list = list(np.concatenate(predictions, axis=1))
            votes = np.asarray(list(
                map(lambda arr: np.bincount(arr).argmax(), argmax_list)
            ))
        if self.voting == 'soft':
            for i, pred in enumerate(predictions):
                pred = list(map(lambda probas: probas * self.weights[i], pred))
                predictions[i] = np.asarray(pred).reshape(nb_preds, self.nb_classes, 1)
            weighted_preds = np.concatenate(predictions, axis=2)
            weighted_avg = np.mean(weighted_preds, axis=2)
            votes = np.argmax(weighted_avg, axis=1)

        return votes
# --------------------------------------------------------------------------

# --------------------  Splitter -------------------------------------------


def getDatasetByUserNumber(dataframe,index):
    return dataframe.query("USER_ID=="+str(index))

def getSampledDatasetByUserNumber(dataframe,index,fraction):
    return dataframe.query("USER_ID=="+str(index)).sample(frac=fraction,replace=True)


def random_partitioning(n_classes, n_classes_per_classifier, stranger_class):

    base_vector = n_classes_per_classifier * [x for x in range(0, n_classes)]
    listOfClusters = []
    for i in range(0, n_classes):
        start = i * n_classes_per_classifier
        end = start + n_classes_per_classifier
        perm = base_vector[start:end]
        perm.append(stranger_class)
        listOfClusters.append(perm)

    return listOfClusters
# --------------------------------------------------------------------------

def clustering_based_partitioning(n_classes, n_classes_per_classifier,X,n_samples,
                                  feature_column,stranger_class, features):

    data = X.iloc[0:n_samples,features]
    classes = X.iloc[0:n_samples,feature_column]

    clf = NearestCentroid()
    clf.fit(data, classes)
    centroids = clf.centroids_

    if n_classes%n_classes_per_classifier!=0:
        raise Exception("The total number of classes and the number of classes per classifier must be congruent.")

    dissim = cosine_similarity(centroids)

    sc = SpectralClustering(n_clusters=int(n_classes/n_classes_per_classifier), affinity='precomputed', n_init=100, assign_labels = 'discretize')
    sc.fit_predict(dissim)

    #ward = AgglomerativeClustering(n_clusters=int(n_classes/n_classes_per_classifier), linkage='ward')
    #ward.fit(centroids)

    clustering = sc
    print("Labels: (label_shape={0})".format(clustering.labels_.shape))

    clusters = dict()
    for label in set(clustering.labels_):
        clusters[label]=[stranger_class]
    id=0
    for label in clustering.labels_:
        clusters[label].append(id)
        id=id+1

    pp.pprint(clusters)
    listOfClusters = []
    for i in range(0, len(clusters.values())):
        listOfClusters.append(list(clusters.values())[i])
    return listOfClusters

# ------- Model creation and training logic ---------------------------------
def createModel(x_train,n_classes,number_of_layers=6,start_dim=1024):
    # creation of the DNN
    model = Sequential()
    # first layer
    model.add(BatchNormalization(input_shape=(x_train.shape[1],)))
    for layer in range(0,number_of_layers):
        model.add(Dense(int(start_dim/math.pow(2,layer)), activation="relu"))
        model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation="softmax"))
    print(model.summary())
    return model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_epoch_end(self, epoch, logs={}):
        print("Epoch:{2} Loss:{0} Acc:{1} ValAcc:{3}".format(logs.get('loss'), logs.get('acc'), epoch,
                                                             logs.get('val_acc')))

def train_classifier(train_dataframe, number_of_classes,row,number_of_epochs=10):

    print("Training classifier for classes:"+str(row)+" ("+str(number_of_classes)+" classes)")
    print(50*"-")
    print("On dataframe: (len:"+str(len(train_dataframe))+")")
    print(train_dataframe.head())
    print(50*"-")

    x = train_dataframe.iloc[:, 0:15].values
    y = train_dataframe.iloc[:, 15].values

    min_max_scaler = MinMaxScaler(feature_range=(-1,1))
    x = min_max_scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)
    y_train = to_categorical(y_train, number_of_classes+1)
    y_test = to_categorical(y_test, number_of_classes+1)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)
    model = createModel(x_train,number_of_classes+1,5,512)

    sgd = SGD(lr=0.05, momentum=0.09, decay=1e-6, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    early_stop_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    # Train the model
    history_callback = model.fit(x_train, y_train, callbacks=[LossHistory(),early_stop_callback], verbose=0,
                                     batch_size=64, epochs=number_of_epochs, validation_data=(x_val, y_val))

    loss_history = history_callback.history["loss"]
    acc_history = history_callback.history["acc"]
    val_loss_history = history_callback.history["val_loss"]
    val_acc_history = history_callback.history["val_acc"]

    print("History: acc={0} val_acc={1} loss={2} val_loss={3}".format(acc_history,val_acc_history,loss_history,
                                                                          val_acc_history))

    score, acc = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
    print("SCORE on Test set:"+str(score))
    print("ACC on Test set:"+str(acc))

    return model,acc,x_test,y_test;

# --------------------------------------------------------------------------


# ----- Global Dataset splitting step --------------------------------------
g_n_classes = 5
g_stranger_class = g_n_classes
g_n_classes_per_classifier = 3
g_n_epochs = 5
selected_partitioning="random"


df = pd.read_csv('./dataset/D2-15F-100C.csv')
users = df.iloc[:, 15]
list_of_users = users.tolist()
list_of_users = list(dict.fromkeys(list_of_users))

for elem in list_of_users:
    index = list_of_users.index(elem)
    df = df.replace({'USER_ID': {elem: index}})

# init
partioning = []

if selected_partitioning=="random":
    # ------------ Partitioning scheme -------------------------------------------
    partioning = random_partitioning(g_n_classes, g_n_classes_per_classifier, g_stranger_class)
    print(20*"-"+"Partitioning:"+20*"-")
    pp.pprint(partioning)
    # ----------------------------------------------------------------------------

if selected_partitioning=="clustering":
    class_ranges = df.iloc[:,15].drop_duplicates()
    pp.pprint(class_ranges)
    last_sample = class_ranges.keys()[g_n_classes]-1
    partioning1 = clustering_based_partitioning(g_n_classes,g_n_classes_per_classifier,df,last_sample,15,g_stranger_class,
                                                [0,1,2,3,4])
    partioning2 = clustering_based_partitioning(g_n_classes,g_n_classes_per_classifier,df,last_sample,15,g_stranger_class,
                                                [5,6,7,8,9,10])
    partioning3 = clustering_based_partitioning(g_n_classes,g_n_classes_per_classifier,df,last_sample,15,g_stranger_class,
                                                [11,12,13,14])
    for perm in partioning1:
        partioning.append(perm)
    for perm in partioning2:
        partioning.append(perm)
    for perm in partioning3:
        partioning.append(perm)

# ------------ Prepare input data form classifiers -------------------------------------------
classifiers = dict()
_id = 0
for cluster_of_classes in partioning:
    dataframe = pd.DataFrame()
    remaining_class_dataframe = pd.DataFrame()

    remaining_classes = set(range(0, g_n_classes+1)) - set(cluster_of_classes)

    stranger_class = cluster_of_classes.pop(len(cluster_of_classes) - 1)

    for elem in cluster_of_classes:
        dataframe = dataframe.append(getDatasetByUserNumber(df,elem))

    for elem in remaining_classes:
        remaining_class_dataframe = remaining_class_dataframe.append(getDatasetByUserNumber(df,elem)) # ,1/len(remaining_classes)))

    remaining_users = remaining_class_dataframe.iloc[:, 15]
    list_of_remaining_users = remaining_users.tolist()
    list_of_remaining_users = list(dict.fromkeys(list_of_remaining_users))

    for elem in list_of_remaining_users:
        remaining_class_dataframe = remaining_class_dataframe.replace({'USER_ID': {elem: stranger_class}})

    dataframe = dataframe.append(remaining_class_dataframe)

    cluster_of_classes.append(stranger_class)

    classifiers[_id] = (cluster_of_classes, dataframe)
    _id = _id + 1

print([(classifiers[_id][0],classifiers[_id][1].tail()) for _id in classifiers])
# ----------------------------------------------------------------------------

# ------------ Train classifiers -------------------------------------------
for id in classifiers:
    print(20*"-"+" Training classifier number {0}/{1} ".format(id+1,len(classifiers))+20*"-")
    cluster_of_classes = classifiers[id][0]
    train_dataframe = classifiers[id][1]
    trained_model, acc, x_test, y_test = train_classifier(train_dataframe, g_n_classes, cluster_of_classes, g_n_epochs)
    classifiers[id]=(cluster_of_classes, train_dataframe, trained_model, acc, x_test, y_test)

pp.pprint("Accuracy of classifiers:")
pp.pprint([classifiers[_id][3] for _id in classifiers ])

# ------------ Test ensemble classifiers using majority voting -------------------------------------------

x_test = classifiers[0][4]
y_test = classifiers[0][5]
for _id in range(1,len(classifiers)):
    x_test = np.append(x_test, classifiers[_id][4],axis=0)
    y_test = np.append(y_test, classifiers[_id][5],axis=0)

models = [classifiers[_id][2] for _id in classifiers]
vm = VotingModel(model_list=models)

x_test = x_test[y_test[:,g_n_classes]==0.0]
y_test = y_test[y_test[:,g_n_classes]==0.0]

results = to_categorical(vm.predict(x_test), g_n_classes+1)

errors = []
outsiders = 0
for _id in range(0, len(results)):
    if (results[_id] == y_test[_id]).all():
        res = 0
    else:
        res = 1
    errors.append(res)

n_errors = 1.0*sum(errors)
n_correct = 1.0*len(y_test)-n_errors
n_total = 1.0*len(y_test)

print("#Errors="+str(n_errors))
print("#Correct="+str(n_correct))
print("#Total="+str(n_total))
print(50*"-")
print("#Outsiders="+str(outsiders))

ensemble_accuracy = 1.0-n_errors/n_total
print("Accuracy="+str(ensemble_accuracy))
