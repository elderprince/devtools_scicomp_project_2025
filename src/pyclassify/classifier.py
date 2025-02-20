import pyclassify.utils as utils

class kNN:
    def __init__(self, k: int):
        if not isinstance(k, int): 
            TypeError("k must be an interger.")
        self.k = k
    
    def _get_k_nearest_neighbors(self, X: list[list[str]], y: list[str], x: list[str]):
        dist_dic = dict()
        for i in range(len(X)): 
            dist = utils.distance(X[i], x)
            dist_dic[i] = (dist, y[i])
        dist_dic_sorted = dict(sorted(dist_dic.items(), key=lambda item: item[1][0]))
        nn_list = list(dist_dic_sorted.items())[:self.k]
        nn = [i[1][1] for i in nn_list]
        return nn
    
    def __call__(self, data: tuple, newPoints: list[list[str]]):
        result = list()
        for i in newPoints: 
            nn = self._get_k_nearest_neighbors(data[0], data[1], i)
            major = utils.majority_vote(nn)
            result.append(major)
        return result