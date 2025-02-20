import pytest
from pyclassify.utils import distance
from pyclassify.utils import majority_vote
from pyclassify.classifier import kNN

def test_distance(): 
    point1 = [1, 0, 0, 1]
    point2 = [0, 1, 1, 0]
    assert distance(point1, point2) == 4

def test_majority_vote(): 
    neighbors = [1, 0, 0, 0]
    assert majority_vote(neighbors) == 0

def test_constructor_of_kNN(): 
    with pytest.raises(TypeError): 
        assert kNN('3'), "k is a string"
        assert kNN(3.5), "k is a float"