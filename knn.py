# load the CIFAR 10 database from torch
import matplotlib.pyplot as plt
import numpy as np

from load_data import *

train_loader, test_loader = load_data()

K_exp = [3, 5, 7, 11]
k_fold = 5


# Build a KNN classifier for CIFAR-10 dataset
class KNN:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X, k):
        y_pred = np.zeros(X.shape[0])

        print("Computing euclidean distances...")
        dists = np.sqrt((X ** 2).sum(axis=1)[:, np.newaxis] + (self.X_train ** 2).sum(axis=1) - 2 * X.dot(self.X_train.T))
        print(f"Distance array shape: {dists.shape}")
        for i in range(len(dists)):
            # sort the distances in increasing order and select the first 'K' rows
            k_nearest_indices = np.argsort(dists[i])[:k]

            # get the labels of the k nearest neighbors
            k_nearest_labels = self.y_train[k_nearest_indices.astype(np.int32)]
            y_pred[i] = np.argmax(np.bincount(k_nearest_labels))
        return y_pred


# train the KNN classifier with 5-fold cross validation
knn = KNN()
accuracy = np.zeros(len(K_exp))
X_train_folds = np.array(np.array_split(train_loader.dataset.data, k_fold))
y_train_folds = np.array(np.array_split(train_loader.dataset.targets, k_fold))

for k in range(len(K_exp)):
    k_exp = K_exp[k]
    acc = 0
    for i in range(k_fold):
        print("K = ", k_exp, "fold = ", i)
        # split the training set into 5 parts
        X_val, y_val = X_train_folds[i], y_train_folds[i]
        X_train, y_train = X_train_folds, y_train_folds

        temp = np.delete(X_train, i, axis=0)
        X_train = np.concatenate(temp, axis=0)
        temp = np.delete(y_train, i, axis=0)
        y_train = np.concatenate(temp, axis=0)

        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_val = np.reshape(X_val, (X_val.shape[0], -1))

        print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
        print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}")
        print("Training the KNN classifier...")

        # train the KNN classifier
        knn.train(X_train, y_train)

        # predict the labels of the test set
        y_pred = knn.predict(X_val, k=k_exp)

        # compute the accuracy
        acc += np.mean(y_pred == y_val)
    accuracy[k] = acc / k_fold
    print("Accuracy for K = ", k_exp, ": ", accuracy[k])
    print("-----------------------------------------------------")

print(accuracy)

# plot the accuracy graph
plt.plot(K_exp, accuracy)
