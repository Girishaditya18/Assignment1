import numpy as np
from load_data import *


class Classifier:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.randn(input_size, output_size)
        self.b = np.zeros((1, output_size))

    def forward(self, x):
        def softmax(x):
            return np.exp(x) / (np.sum(np.exp(x), axis=1, keepdims=True) + 1e-8)

        z = np.dot(x, self.W) + self.b
        y = softmax(z)
        return y

    def loss_fn(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def gradient_descent(self, loss, learning_rate):
        self.W -= learning_rate * loss
        self.b -= learning_rate * loss

    def predict(self, x):
        return self.forward(x)


if __name__ == "__main__":
    train_loader, test_loader = load_data(batch_size=64)
    epochs = 100
    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    train_acc = np.zeros(epochs)
    val_acc = np.zeros(epochs)
    classifier = Classifier(32 * 32 * 3, 10)
    learning_rate = 0.03

    for it in range(epochs):

        print(f"Epoch {it + 1}/{epochs}")
        print("\nTraining...")

        temp_loss = 0
        temp_acc = 0
        for i, (x_train, y_train) in enumerate(train_loader):
            x = np.reshape(x_train.numpy(), (x_train.shape[0], -1))
            y = np.reshape(y_train.numpy(), (y_train.shape[0], -1))

            y_pred = classifier.forward(x)
            y_pred = np.expand_dims(np.argmax(y_pred, axis=1), axis=1)
            loss = classifier.loss_fn(y, y_pred)
            classifier.gradient_descent(loss, learning_rate)

            temp_loss += loss
            temp_acc += np.sum(y_pred == y)
            print(f"\r{i + 1}/{len(train_loader)}", end="")

        print()
        train_loss[it] = temp_loss / len(train_loader)
        train_acc[it] = temp_acc / len(train_loader)

        temp_loss = 0
        temp_acc = 0
        print("\nValidating...")
        for i, (x_test, y_test) in enumerate(test_loader):
            x = np.reshape(x_test.numpy(), (x_test.shape[0], -1))
            y = np.reshape(y_test.numpy(), (y_test.shape[0], -1))

            y_pred = classifier.forward(x)
            y_pred = np.expand_dims(np.argmax(y_pred, axis=1), axis=1)
            loss = classifier.loss_fn(y, y_pred)

            temp_loss += loss
            temp_acc += np.sum(y_pred == y)
            print(f"\r{i + 1}/{len(test_loader)}", end="")

        print()
        val_loss[it] = temp_loss / len(test_loader)
        val_acc[it] = temp_acc / len(test_loader)
        print(f"Train loss: {train_loss[it]}, val loss: {val_loss[it]}")
        print(f"Train acc: {train_acc[it]}, val acc: {val_acc[it]}")

    print(f"train loss: {np.mean(train_loss)}, val loss: {np.mean(val_loss)}")
    print(f"train acc: {np.mean(train_acc)}, val acc: {np.mean(val_acc)}")
