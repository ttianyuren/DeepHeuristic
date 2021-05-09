from deep_heuristic.visualize_training_data import split_data
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import pickle as pk
# import pandas as pd
import matplotlib.pyplot as plt

file_reach = '../training_data/reach.pk'
with open(file_reach, 'rb') as f:
    all_data = pk.load(f)

train_data, train_labels, eval_data, eval_labels = split_data(all_data, test_size=0.1)
# 1 layer: 600:97.37 / 10:95.87 / 30:97.55 / 50:97.65 / 100:97.37 / 1000:97.35
# 2 layers: 10:96.66 / 100:96.86 / 500:97.88 / 1000:97.92
# 3 layers: 10:96.41 / 100:97.32 / 500:96.41 / 30.60.20:97.12 /
for nodes in range(100, 301, 50):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(nodes, nodes), random_state=1)
    clf.fit(train_data, train_labels+0)
    print(nodes, ": \n", confusion_matrix(eval_labels, clf.predict(eval_data)))
    print("##################################")
    # plot_confusion_matrix(clf, eval_data, eval_labels+0)
    # plt.show()
print("data:", len(train_data))
