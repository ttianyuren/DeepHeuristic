from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from deep_heuristic.visualize_training_data import split_data
import pickle as pk
import matplotlib.pyplot as plt


file_reach = '../training_data/reach.pk'
# all_data = read_pickle(file_reach)
with open(file_reach, 'rb') as f:
    all_data = pickle.load(f)

train_data, train_labels, eval_data, eval_labels = split_data(all_data, test_size=0.1, num_of_param=3)

clf = SVC(gamma='auto')
clf.fit(train_data, train_labels)
plot_confusion_matrix(clf, eval_data, eval_labels)
plt.show()
