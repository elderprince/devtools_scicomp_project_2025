import numpy as np
import pyclassify.utils as utils
from line_profiler import profile

class kNN:
    def __init__(self, k: int, backend='plain'):
        if not isinstance(k, int): 
            TypeError("k must be an interger.")
        self.k = k
        self.backend = backend
    
    @profile
    def _get_k_nearest_neighbors(self, X: list[list[str]], y: list[str], x: list[str]):
        dist_dic = dict()
        for i in range(len(X)): 
            point1 = X[i]
            point2 = x
            if self.backend == 'plain': 
                dist = utils.distance(point1, point2)
            elif self.backend == 'numpy': 
                point1 = np.array([float(a) for a in point1])
                point2 = np.array([float(b) for b in point2])
                dist = utils.distance_numpy(point1, point2)
            elif self.backend == 'numba': 
                point1 = np.array([float(a) for a in point1])
                point2 = np.array([float(b) for b in point2])
                dist = utils.distance_numba(point1, point2)
            else: 
                ValueError("'backend' must be set to 'plain', 'numpy', or 'numba'.")
            dist_dic[i] = (float(dist), y[i])
        dist_dic_sorted = dict(sorted(dist_dic.items(), key=lambda item: item[1][0]))
        nn_list = list(dist_dic_sorted.items())[:self.k]
        nn = [i[1][1] for i in nn_list]
        return nn
    
    @profile
    def __call__(self, data: tuple, newPoints: list[list[str]]):
        result = list()
        for i in newPoints: 
            nn = self._get_k_nearest_neighbors(data[0], data[1], i)
            major = utils.majority_vote(nn)
            result.append(major)
        return result