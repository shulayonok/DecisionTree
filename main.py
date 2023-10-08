import random
import titanic
import digits
import iris
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_diabetes
from DecisionTree import Calssification_Tree
from DecisionTree import Regression_Tree

"""
# Считывание данных (титаник)
X = titanic.read()
X, target = titanic.preparing(X)

# Разделение выборки (титаник)
N = len(X)
train_len = int(N * 0.8)
validation_len = int(N * 0.1)
test_len = int(N * 0.1)
Train, Validation, Test, T_Train, T_Validation, T_Test = titanic.divide(X, target, train_len, validation_len, test_len)
"""

"""
# Считывание данных (ирисы)
X, target = load_iris(return_X_y=True)
x1, target1 = X[0], target[0]
x2, target2 = X[1], target[1]
X, target = X[2:], target[2:]

# Разделение выборки (ирисы)
N = len(X)
train_len = int(N * 0.8)
validation_len = int(N * 0.1)
test_len = int(N * 0.1)
Train, Validation, Test, T_Train, T_Validation, T_Test = iris.splitting(X, target, train_len, validation_len, test_len)
"""

# Считывание данных (цифры)
Digits = load_digits()
N = len(Digits["data"])

# Длины выборок
train = int(N * 0.7)  # 80%
validation = int(N * 0.2)  # 10%
test = int(N * 0.1)  # 10%

# Стандартизация, разделение
data = digits.standardisation(Digits["data"])
Train, Validation, Test, T_Train, T_Validation, T_Test = digits.splitting(data, Digits["target"], train, validation, test)


# Валидация гиперпараметров
classifiers = []
accuracy = []
for i in range(10):
    classifier = Calssification_Tree(max_depth=random.randint(5, 25), min_quantity=random.randint(5, 50))
    classifier.fit(Train, T_Train)
    classifiers.append(classifier)
    prediction = classifier.prediction(Validation)
    acc = np.sum(prediction == T_Validation) / len(T_Validation)
    accuracy.append(acc)
    print(acc)

print(f"Best accuracy: {max(accuracy)}")
best_tree = classifiers[accuracy.index(max(accuracy))]
print(f"Best tree: max_depth = {best_tree.max_depth}, min_quantity = {best_tree.min_quantity}")
print()

# Точность на тестовой выборке
prediction = best_tree.prediction(Test)
acc = np.sum(prediction == T_Test) / len(T_Test)
print(f"Accuracy on test set: {acc}\n")

"""
# Вектора уверенности (титаник)
x = np.array([3, 0, 22, 1, 0, 7.25], dtype=float)  # Class 0
t0, t1 = best_tree.prediction(x, option=True)
print(t0, t1)

print()

x = np.array([1, 1, 38, 1, 0, 71.2833], dtype=float)  # Class 1
t0, t1 = best_tree.prediction(x, option=True)
print(t0, t1)
"""

"""
# Вектора уверенности (ирисы)
print(f"Real class: {target1}")
print(best_tree.one_prediction(x1, best_tree.tree))

print()

print(f"Real class: {target2}")
print(best_tree.one_prediction(x2, best_tree.tree))
"""

"""
# Задача регрессии
X, target = load_diabetes(return_X_y=True)

# Разделение выборки (диабет)
N = len(X)
train_len = int(N * 0.8)
validation_len = int(N * 0.1)
test_len = int(N * 0.1)
Train, Validation, Test, T_Train, T_Validation, T_Test = diabetes.splitting(X, target, train_len, validation_len, test_len)

# Валидация гиперпараметров
classifiers = []
accuracy = []
for i in range(10):
    classifier = Regression_Tree(max_depth=random.randint(8, 18), min_quantity=random.randint(8, 18))
    classifier.fit(Train, T_Train)
    classifiers.append(classifier)
    prediction = classifier.prediction(Validation)
    acc = 1 - np.sum((prediction - T_Validation)**2) / np.sum((T_Validation.mean() - T_Validation)**2)
    accuracy.append(acc)
    print(acc)

print(f"Best accuracy: {max(accuracy)}")
best_tree = classifiers[accuracy.index(max(accuracy))]
print(f"Best tree: max_depth = {best_tree.max_depth}, min_quantity = {best_tree.min_quantity}")
print()

# Точность на тестовой выборке
prediction = best_tree.prediction(Test)
acc = 1 - np.sum((prediction - T_Test)**2) / np.sum((T_Test.mean() - T_Test)**2)
print(f"Accuracy on test set: {acc}\n")
"""

"""
test_score = []
n_trees = []

for i in tqdm(range(10, 50, 3)):
    classifier = Calssification_Tree(min_quantity=i)
    classifier.fit(Train, T_Train)
    n_trees.append(i)
    prediction = classifier.prediction(Test)
    acc = np.sum(prediction == T_Test) / len(T_Test)
    test_score.append(acc)

plt.plot(n_trees, test_score)
plt.show()
"""
