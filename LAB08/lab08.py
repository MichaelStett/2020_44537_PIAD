import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


# https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
def wiPCA(points):
    mean_ = np.mean(points.T, axis=1)
    
    C = points - mean_
    V = np.cov(C.T)
    
    explained_variance_, components_ = np.linalg.eigh(V, UPLO='L')

    data = components_.T.dot(C.T).T
    
    return data, mean_, components_, explained_variance_

# 1.
### wiPCA - Brak rzutu na pierwszą składową
rng = np.random.RandomState(1)
points = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

data, mean_, components_, explained_variance_ = wiPCA(points)

print("-------------")
print(f'explained_variance_ : {explained_variance_}')
print(f'components_ : {components_}')
print(f'mean_ : {mean_}')

for length, vector in zip(explained_variance_, components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(mean_, mean_ + v)

plt.title('wiPCA')
plt.scatter(points[:, 0], points[:, 1], s=8, c='g')
#plt.scatter(data[:, 0], data[:, 1], s=9, c='r')
plt.axis('equal');
plt.show()

### sklearn.PCA
pca = PCA(n_components=2)
print(pca)
pca.fit(points)

print("-------------")
print(f'explained_variance_ : {pca.explained_variance_}')
print(f'components_ : {pca.components_}')
print(f'mean_ : {pca.mean_}')

for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)

pca = PCA(n_components=1)
pca.fit(points)
point_pca = pca.transform(points)
points_new = pca.inverse_transform(point_pca)

plt.title('sklearn.PCA')
plt.scatter(points[:, 0], points[:, 1], s=8, c='g')
plt.scatter(points_new[:, 0], points_new[:, 1], s=9, c='r')
plt.axis('equal');
plt.show()

# 2. iris - działa
iris = datasets.load_iris()
points = iris.data

#plt.scatter(points[:, 0], points[:, 1], s=8, c=iris.target)
#plt.show()

data, mean_, components_, explained_variance_ = wiPCA(points)

print("-------------")
print(f'explained_variance_ : {explained_variance_}')
print(f'components_ : {components_}')
print(f'mean_ : {mean_}')

plt.title('iris')
plt.scatter(data[:, 0], data[:, 1], s=9,c=iris.target)
plt.axis('equal');
plt.show()

# 3. digits - działa
digits = datasets.load_digits()
points = digits.data

#plt.scatter(points[:, 0], points[:, 1], s=8, c=digits.target)
#plt.show()

data, mean_, components_, explained_variance_ = wiPCA(points)

print("-------------")
print(f'explained_variance_ : {explained_variance_}')
print(f'components_ : {components_}')
print(f'mean_ : {mean_}')

plt.title('digits')
plt.scatter(data[:, 0], data[:, 1], s=9, c=digits.target)
plt.colorbar()
plt.axis('equal');
plt.show()
