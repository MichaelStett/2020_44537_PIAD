import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from operator import itemgetter

def distance(p1, p2):
    p1, p2 = np.array(p1), np.array(p2)

    return np.sqrt(sum(np.subtract(p1, p2) ** 2))

class KNN:
    def __init__(self, n_neighbors = 1, use_KDTree = False):
        """
            n_neighbors - liczba sąsiadów,
            use_KDTree - definiuje czy należy korzystać kD-drzew
        """
        self.n_neighbors = n_neighbors
        self.use_KDTree = use_KDTree


    def fit(self, X, y):
        """
        Metoda do uczenia modelu.
            X - dane wejściowe
            y - atrybuty decyzyjne/zmienna zależna
        """

        assert len(X) == len(y)

        self.data = [ ([coords[0], coords[1]], label) for coords, label in zip(X, y)]


    def predict(self, X):
        """
        Metoda do dokonywania predykcji.
        """

        predicts = []
        for coords in X:
            pred_point = [coords[0], coords[1]]

            distances = []

            for point, label in self.data:
                distances.append((distance(pred_point, point), label))

            distances = sorted(distances)[:self.n_neighbors]

            predicts.append(list(max(distances))[1])

        return predicts


    def score(self, X, y):
        """
        Metoda do dokonywania predykcji.
        Metoda powinna zwracać błąd średniokwadratowy dla zadania regresji lub procentową dokładność w zadaniu klasyfikacji.

        """
        pass

iris = datasets.load_iris()
data = iris.data
labels = iris.target

knn = KNN(n_neighbors=1)
knn.fit(data, labels)

print(type(knn.data))
print(knn.data)
print(knn.predict(data))


print(knn.data[0]) # points
print(knn.data[1]) # labels
plt.plot(knn.data[0][0], knn.data[0][1])


# 3.

