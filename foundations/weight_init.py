import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        res = x
        # weights = np.array(weights)
        # bias = np.array(bias)
        for i in range(len(weights)):
            print(weights[i].shape)
        for i in range(len(weights)):
            print("res shape," , res.shape)
            res = res@weights[i] +biases[i]
            if i < len(weights) - 1:
                print(res.shape)
                res = res*(res>0)
        return np.round(res,5)
        
