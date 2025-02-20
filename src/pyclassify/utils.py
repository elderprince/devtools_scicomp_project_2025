import os
import yaml

def distance(point1: list[str], point2: list[str]): 
    dis_squ = 0
    for i in range(len(point1)): 
        dis_diff = float(point1[i]) - float(point2[i])
        diff_squ = dis_diff * dis_diff
        dis_squ += diff_squ
    return dis_squ

def majority_vote(neighbors: list[str]): 
    neighbor_count = dict((x,neighbors.count(x)) for x in set(neighbors))
    neighbor_sorted = dict(sorted(neighbor_count.items(), key=lambda item: item[1], reverse=True))
    majority = neighbor_sorted.popitem()[0]
    return majority

def read_config(file):
   filepath = os.path.abspath(f'{file}.yaml')
   with open(filepath, 'r') as stream:
      kwargs = yaml.safe_load(stream)
   return kwargs