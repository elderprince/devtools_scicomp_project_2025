from pyclassify.utils import read_config
from pyclassify.utils import read_file
from pyclassify.classifier import kNN
import random

kwargs = read_config('experiments/config_spambase')

k = int(kwargs['k'])
dataset = kwargs['dataset']
backend = kwargs['backend']

data = read_file(dataset)

train_rate = int(0.8 * len(data))

random.shuffle(data)

train_data = data[:train_rate]
test_data = data[train_rate:]

train_points = list()
train_labels = list()

test_points = list()
test_labels = list()

for i in train_data: 
    train_points.append(i[:-1])
    train_labels.append(i[-1])

for i in test_data: 
    test_points.append(i[:-1])
    test_labels.append(i[-1])

model = kNN(k, backend)
pred_labels = model((train_points, train_labels), test_points)
accuracy = len([test_labels[i] for i in range(0, len(test_labels)) if test_labels[i] == pred_labels[i]]) / len(test_labels)

print('The accuracy of this kNN classifier is ' + str(accuracy))