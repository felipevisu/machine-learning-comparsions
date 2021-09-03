import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from dataset import load_data

X, y = load_data()

tree = DecisionTreeClassifier(
    criterion='gini', 
    min_samples_leaf=1, 
    min_samples_split=6,
    random_state=4,
    splitter='random'
)
tree_scores = cross_val_score(tree, X, y, cv=10, scoring='accuracy')

knn = KNeighborsClassifier(
    algorithm='auto',
    n_neighbors=15,
    weights='distance'
)
knn_scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')

svm = SVC(
    C=5,
    decision_function_shape='ovo',
    gamma='auto'
)
svm_scores = cross_val_score(svm, X, y, cv=10, scoring='accuracy')
 
data = [tree_scores, knn_scores, svm_scores]
 
fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(16)
bp = ax.boxplot(data)
ax.set_xticklabels(['Tree', 'KNN','SVM'])
 
plt.rcParams.update({'font.size': 16})
plt.title("Comparação entre os métodos")
plt.show()