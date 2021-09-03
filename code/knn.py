from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from statistics import variance, stdev
from dataset import load_data

X, y = load_data()

param_grid = dict(
    n_neighbors=range(1, 31), 
    weights=['uniform', 'distance'],
    algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'],
)
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)

print('\nExperimento fatorial:\n')
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print("\nMelhor calibragem:", grid.best_params_)
print("Melhor acurácia:", grid.best_score_)

knn = KNeighborsClassifier(**grid.best_params_)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print("\nMédia de acurácia:", scores.mean())
print("Variância:", variance(scores))
print("Desvio padrão:", stdev(scores))
print()