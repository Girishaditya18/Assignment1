import numpy as np
from load_data import *

train_loader, test_loader = load_data()


class KMeans:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.loss = []
        self.accuracy = []
        self.centroids = []

    def fit(self, X, y):
        # initialize centroids with random samples from the training set.
        for _ in range(self.k):
            self.centroids.append(X[np.random.randint(0, X.shape[0])])

        distances = np.zeros((X.shape[0], self.k))
        step = 0
        while step < self.max_iter:
            for i, k in enumerate(self.centroids):
                distances[:, i] = np.linalg.norm(X - k, axis=1)

            # assign each sample to the closest centroid
            labels = np.argmin(distances, axis=1)

            # update centroids
            for i in range(self.k):
                self.centroids[i] = np.mean(X[labels == i], axis=0)

            # print loss
            self.loss.append(np.mean(distances))
            self.accuracy.append(np.mean(labels == y))
            print(f"iteration: {step} ==> loss: {self.loss[-1]}, accuracy: {self.accuracy[-1]}")

            step += 1

    def predict(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for i, k in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - k, axis=1)

        return np.argmin(distances, axis=1)


k_fold = 5
X_train_folds = np.array(np.array_split(train_loader.dataset.data, k_fold))
y_train_folds = np.array(np.array_split(train_loader.dataset.targets, k_fold))

for i in range(k_fold):
    print("fold = ", i)
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
    print("Training the K-Means classifier...")

    # train the KNN classifier
    kmeans = KMeans(k=10)
    kmeans.fit(X_train, y_train)
    y_pred = kmeans.predict(X_val)

    # compute accuracy
    accuracy = np.mean(y_pred == y_val)
