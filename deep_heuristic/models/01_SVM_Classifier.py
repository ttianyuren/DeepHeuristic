from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from deep_heuristic.nn_utils import split_data
import pickle as pk
import matplotlib.pyplot as plt

train_data, train_labels, eval_data, eval_labels = split_data(load_workspace(), test_size=0.1, num_of_param=3)

clf = SVC(gamma='auto')
clf.fit(train_data, train_labels)
plot_confusion_matrix(clf, eval_data, eval_labels)
plt.show()
