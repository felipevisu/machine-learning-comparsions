from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from statistics import variance, stdev
from dataset import load_data

X, y = load_data()

param_grid = dict(
    criterion=["gini", "entropy"], 
    splitter=["best", "random"],
    random_state=[0, 4, 8],
    min_samples_split=[2, 4, 6],
    min_samples_leaf=[1, 4]
)
grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=10, scoring='accuracy')
grid.fit(X, y)

print('\nExperimento fatorial:\n')
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print("\nMelhor calibragem:", grid.best_params_)
print("Melhor acurácia:", grid.best_score_)

tree = DecisionTreeClassifier(**grid.best_params_)
scores = cross_val_score(tree, X, y, cv=10, scoring='accuracy')
print("\nMédia de acurácia:", scores.mean())
print("Variância:", variance(scores))
print("Desvio padrão:", stdev(scores))
print()