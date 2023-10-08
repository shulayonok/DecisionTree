import numpy as np


# Информация об узлах дерева (характеристика, значение характеристики, левая часть дерева, правая часть дерева и
# значение в конечном узле
class Node:
    def __init__(self, feature=None, threshold=None, leftNode=None, rightNode=None, value=None, t=None):
        self.feature = feature
        self.threshold = threshold
        self.leftNode = leftNode
        self.rightNode = rightNode
        self.value = value
        self.t = t

    def is_terminal(self):
        return self.value is not None


class Calssification_Tree:
    def __init__(self, max_depth=10, min_quantity=10):
        self.max_depth = max_depth
        self.min_quantity = min_quantity
        self.tree = None  # Построенное дерево

    # Обучение алгоритма
    def fit(self, X, target):
        self.tree = self.grow_tree(X, target)

    # Предсказание
    def prediction(self, X, option=False):
        if option:
            return self.confidence(X, self.tree)
        return np.array([self.one_prediction(x, self.tree) for x in X])

    # Расчёт энтропии
    def entropy(self, target):
        unique, count = np.unique(target, return_counts=True)
        p = count / len(target)
        entropy = -np.sum([pi * np.log2(pi) for pi in p if pi > 0])
        gini = np.sum([pi * (1 - pi) for pi in p])
        return gini

    # Возращает наиболее частую метку
    def argmaxT(self, target):
        unique, count = np.unique(target, return_counts=True)
        return unique[np.argmax(count)]

    # Целева функция (прирост информации)
    def information_gain(self, X_column, target, threshold):
        if len(np.unique(target)) == 1:
            return 0

        # Информативность в родителе
        n = len(target)
        parent = self.entropy(target)

        left_indexes = np.argwhere(X_column <= threshold).T[0]
        right_indexes = np.argwhere(X_column > threshold).T[0]

        # Информативность в потомках
        entopy_left, n_left = self.entropy(target[left_indexes]), len(left_indexes)
        entopy_right, n_right = self.entropy(target[right_indexes]), len(right_indexes)

        child = (n_left / n) * entopy_left + (n_right / n) * entopy_right

        return parent - child

    # Лучшее разбиение
    def best_dividing(self, X, target):
        best_feature, best_threshold = None, None
        best_gain = -1

        # Проходим по всем характеристикам и по всем порогам
        for i in range(X.shape[1]):
            thresholds = np.unique(X[:, i])
            for threshold in thresholds:
                gain = self.information_gain(X[:, i], target, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    best_threshold = threshold

        return best_feature, best_threshold

    # Рекурсивный алгоритм построения дерева
    def grow_tree(self, X, target, depth=0):
        # Кол-во элементов
        quantity = len(target)
        # Кол-во меток
        labels = len(np.unique(target))

        # Критерий остановки алгоритма
        if (depth >= self.max_depth or quantity <= self.min_quantity) and labels != 1:
            unique, count = np.unique(target, return_counts=True)
            print(depth)
            return Node(value=self.argmaxT(target), t=(count[0] / len(target), count[1] / len(target)))
        elif labels == 1:
            value = self.argmaxT(target)
            if value == 0:
                print(depth)
                return Node(value=value, t=(1, 0))
            else:
                print(depth)
                return Node(value=value, t=(0, 1))

        best_feature, best_threshold = self.best_dividing(X, target)

        left_indexes = np.argwhere(X[:, best_feature] <= best_threshold).T[0]
        right_indexes = np.argwhere(X[:, best_feature] > best_threshold).T[0]

        leftNode = self.grow_tree(X[left_indexes, :], target[left_indexes], depth + 1)
        rightNode = self.grow_tree(X[right_indexes, :], target[right_indexes], depth + 1)

        return Node(best_feature, best_threshold, leftNode, rightNode)

    # Рекурсивный алгоритм предсказания
    def one_prediction(self, x, tree):
        # Критерий остановки
        if tree.is_terminal():
            return tree.value

        if x[tree.feature] < tree.threshold:
            return self.one_prediction(x, tree.leftNode)
        return self.one_prediction(x, tree.rightNode)

    # Рекурсивный алгоритм предсказания
    def confidence(self, x, tree):
        # Критерий остановки
        if tree.is_terminal():
            return tree.t

        if x[tree.feature] < tree.threshold:
            return self.confidence(x, tree.leftNode)
        return self.confidence(x, tree.rightNode)


class Regression_Tree:
    def __init__(self, max_depth=10, min_quantity=15):
        self.max_depth = max_depth
        self.min_quantity = min_quantity
        self.tree = None  # Построенное дерево

    # Обучение алгоритма
    def fit(self, X, target):
        self.tree = self.grow_tree(X, target)

    # Предсказание
    def prediction(self, X):
        return np.array([self.one_prediction(x, self.tree) for x in X])

    # Расчёт среднеквадр. отклонения (абсолютного отклонения)
    def entropy(self, target):
        const = np.sum(target) / len(target)
        mse = np.sum((const - target) ** 2) / len(target)
        mae = np.sum(np.abs(const - target)) / len(target)
        return mse

    # Константный прогноз
    def argmaxT(self, target):
        return np.sum(target) / len(target)

    # Целева функция (прирост информации)
    def information_gain(self, X_column, target, threshold):
        if len(np.unique(target)) == 1:
            return 0

        # Информативность в родителе
        n = len(target)
        parent = self.entropy(target)

        left_indexes = np.argwhere(X_column <= threshold).T[0]
        right_indexes = np.argwhere(X_column > threshold).T[0]

        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return 0

        # Информативность в потомках
        entopy_left, n_left = self.entropy(target[left_indexes]), len(left_indexes)
        entopy_right, n_right = self.entropy(target[right_indexes]), len(right_indexes)

        child = (n_left / n) * entopy_left + (n_right / n) * entopy_right

        return parent - child

    # Лучшее разбиение
    def best_dividing(self, X, target):
        best_feature, best_threshold = None, None
        best_gain = -1

        # Проходим по всем характеристикам и по всем порогам
        for i in range(X.shape[1]):
            thresholds = np.unique(X[:, i])
            for threshold in thresholds:
                gain = self.information_gain(X[:, i], target, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    best_threshold = threshold

        return best_feature, best_threshold

    # Рекурсивный алгоритм построения дерева
    def grow_tree(self, X, target, depth=0):
        # Кол-во элементов
        quantity = len(target)
        # Кол-во меток
        labels = len(np.unique(target))

        # Критерий остановки алгоритма
        if depth >= self.max_depth or quantity <= self.min_quantity or labels == 1:
            return Node(value=self.argmaxT(target))

        best_feature, best_threshold = self.best_dividing(X, target)

        left_indexes = np.argwhere(X[:, best_feature] <= best_threshold).T[0]
        right_indexes = np.argwhere(X[:, best_feature] > best_threshold).T[0]

        leftNode = self.grow_tree(X[left_indexes, :], target[left_indexes], depth + 1)
        rightNode = self.grow_tree(X[right_indexes, :], target[right_indexes], depth + 1)

        return Node(best_feature, best_threshold, leftNode, rightNode)

    # Рекурсивный алгоритм предсказания
    def one_prediction(self, x, tree):
        # Критерий остановки
        if tree.is_terminal():
            return tree.value

        if x[tree.feature] < tree.threshold:
            return self.one_prediction(x, tree.leftNode)
        return self.one_prediction(x, tree.rightNode)
