import numpy as np


class LinearRegression:
    w: np.ndarray
    b: float

    def __init__(self):
        # raise NotImplementedError()
        self.w = np.array([])
        self.b = 0.0

    def fit(self, X, y):
        n, p = X.shape
        X = np.c_[np.ones((n, 1)), X]
        w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        self.w = w[1:]
        self.b = w[0]
        # raise NotImplementedError()

    def predict(self, X):
        # raise NotImplementedError()
        yhat = X @ self.w + self.b
        return yhat


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
            self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        n, d = X.shape
        if self.w is None:
            self.w = np.zeros((d, 1))
            self.b = 0.0

        w = np.concatenate([self.b, self.w])
        X = np.c_[np.ones((n, 1)), X]
        for _ in range(epochs):
            y_hat = X @ w
            gradient = np.sum(np.dot((y_hat - y).T, X), axis=0) * (1 / len(y_hat))
            gradient = np.reshape(gradient, (-1, 1))
            w = w - lr * gradient
        self.w = w[1:]
        self.b = w[0]
        # raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        yhat = X @ self.w + self.b
        return yhat
        # raise NotImplementedError()
