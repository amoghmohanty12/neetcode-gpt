import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        z1 = x@W1.T + b1
        rel1 = z1*(z1>0)
        y_hat = rel1@W2.T + b2
        loss = np.mean((y_hat-y_true)**2)
        dl_dy = 2*(y_hat - y_true) / y_true.size
        dy_dw2 = np.outer(dl_dy,rel1)
        dy_drel = dl_dy @ W2* (z1 > 0)
        dy_dw1 = np.outer(dy_drel, x)
        dl_db2 = dl_dy*1
        # dl_drel = np.outer(d,W2)*(z1>0)
        dl_db1 = dy_drel*1
        dicti = {'loss':np.round(loss,4),'dW1':np.round(dy_dw1,4),'db1':np.round(dl_db1,4),
        'dW2':np.round(dy_dw2,4),'db2':np.round(dl_db2,4)}
        return dicti

