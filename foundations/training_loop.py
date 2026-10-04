import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        n, d = X.shape

        w = np.zeros(d)   # shape: (d,)
        # w = np.zeros((1,X.shape[1]))
        print(w.shape)
        print(X.shape)
        b=0
        for i in range(epochs):
            pred = X @ w+b
            loss = (1/len(X))*np.sum((pred-y)**2)
            w = w-lr*2/len(X)*X.T@(pred-y)
            b = b-2*lr/len(X)*np.sum(pred-y)
        return(np.round(w,5),np.round(b,5))

