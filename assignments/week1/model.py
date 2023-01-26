import numpy as np


class LinearRegression:
    """
        A linear regression model that uses closed form solution to fit the model.
    """
    w: np.ndarray
    b: float

    def __init__(self):
        # raise NotImplementedError()
        self.w = np.array([])
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Predict the weights and bias for the given input.

        Arguments:
            X (np.ndarray): The input data
            y (np.ndarray): labels

        """
        n, p = X.shape
        X = np.c_[np.ones((n, 1)), X]
        w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        self.w = w[1:]
        self.b = w[0]
        # raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data

        Returns:
            np.ndarray: The predicted output.

        """
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
        """
        Predict the weights and bias for the given input.

        Arguments:
            X (np.ndarray): The input data
            y (np.ndarray): labels
            lr (float): learning rate
            epochs (int)

        """
        n, d = X.shape
        w = np.random.randn(n + 1)

        X = np.c_[np.ones((n, 1)), X]
        for _ in range(epochs):
            y_hat = np.dot(X, w.T)
            gradient = (y_hat-y) @ X / n
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
