import os
import csv
import yaml
from numba.pycc import CC
from line_profiler import profile

@profile
def distance(point1: list[str], point2: list[str]): 
    dis_squ = 0
    for i in range(len(point1)): 
        dis_diff = float(point1[i]) - float(point2[i])
        diff_squ = dis_diff * dis_diff
        dis_squ += diff_squ
    return dis_squ

@profile
def distance_numpy(point1: list, point2: list): 
    dis_squ = (point1 - point2) ** 2
    dis_squ_sum = dis_squ.sum()
    return dis_squ_sum

cc = CC('module')
@cc.export('distance_numba', 'f8(f8[:], f8[:])')
@profile
def distance_numba(point1: list, point2: list): 
    dis_squ_sum = 0
    for i in range(point1.shape[0]): 
        dis_squ_sum += (point1[i] - point2[i]) ** 2
    return dis_squ_sum

@profile
def majority_vote(neighbors: list[str]): 
    neighbor_count = dict((x,neighbors.count(x)) for x in set(neighbors))
    neighbor_sorted = dict(sorted(neighbor_count.items(), key=lambda item: item[1], reverse=True))
    majority = neighbor_sorted.popitem()[0]
    return majority

@profile
def read_config(file):
   filepath = os.path.abspath(f'{file}.yaml')
   with open(filepath, 'r') as stream:
      kwargs = yaml.safe_load(stream)
   return kwargs

@profile
def read_file(file): 
    with open(file, mode ='r') as file:
        csvFile = csv.reader(file)
        data = [line for line in csvFile]
    return data

if __name__ == "__main__":
    cc.compile()