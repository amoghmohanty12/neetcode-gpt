import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        summ = 0
        maxx = max(z)
        for i in z:
            summ = summ+math.exp(i-maxx)
        res = []
        for i in z:
            res.append(np.round((math.exp(i-maxx))/(summ),4))
        return res
