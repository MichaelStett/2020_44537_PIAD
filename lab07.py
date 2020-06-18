import scipy 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from PIL import Image

from matplotlib import style
from sklearn import datasets
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import jaccard_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import ConvexHull

style.use('ggplot')

COLORS = [
    'r', 'g', 'b', 'c', 'm', 'y'
]


def find_perm(clusters, Y_real, Y_pred):
    perm = []
    for i in range(clusters):
        idx = Y_pred == i
        new_label = scipy.stats.mode(Y_real[idx])[0][0]
        perm.append(new_label)
    return [perm[label] for label in Y_pred]


def xplot(type, X, Y, Y2 = [], classify=False, convex=False, title = 'no title'):
    if type == '2D':
        plt.figure()

        if classify == True:
            Y_diff = [1 if y == y_fit else 0 for y, y_fit in zip(Y, Y2)]

            for i, x, y in zip(range(X.shape[0]), X, Y_diff):
                plt.plot(x[0], x[1], 'o', color=COLORS[y])
        else:
            for i, x, y in zip(range(X.shape[0]), X, Y):
                plt.plot(x[0], x[1], 'o', color=COLORS[y])

            # Convex Hull
            if convex:
                for i, y in enumerate(np.unique(Y)):
                    points = X[Y == y]
                    hull = ConvexHull(points)

                    for simplex in hull.simplices:
                        plt.plot(points[simplex, 0], points[simplex, 1], '-', color='k')
        plt.title(title)
        plt.show();
    elif type == '3D':
        fig = plt.figure();
        
        ax = fig.add_subplot(111, projection='3d')

        if classify == True:
            Y_diff = [1 if y == y_fit else 0 for y, y_fit in zip(Y, Y2)]

            for i, x, y in zip(range(X.shape[0]), X, Y_diff):
                ax.scatter(x[0], x[1], x[2], 'o', color=COLORS[y])

        else:
            for i, x, y in zip(range(X.shape[0]), X, Y):
                ax.scatter(x[0], x[1], x[2], 'o', color=COLORS[y])

        plt.title(title)
        plt.show();

### Klasteryzacja ###

iris = datasets.load_iris()
X = iris.data
Y = iris.target

# metoda najbliższego sąsiedztwa - single
# metoda średnich połączeń       - average
# metoda najdalszych połączeń    - complete
# metoda Warda                   - ward

#(clust_S, clust_A, clust_C, clust_W) = (
#    AgglomerativeClustering(n_clusters=3, linkage='single'  ).fit(X),
#    AgglomerativeClustering(n_clusters=3, linkage='average' ).fit(X),
#    AgglomerativeClustering(n_clusters=3, linkage='complete').fit(X),
#    AgglomerativeClustering(n_clusters=3, linkage='ward'    ).fit(X),
#)

#print(f'single:   {clust_S.labels_}'); print()
#print(f'average:  {clust_A.labels_}'); print()
#print(f'complete: {clust_C.labels_}'); print()
#print(f'ward:     {clust_W.labels_}'); print()

Y_real = Y

#(Y_pred_S, Y_pred_A, Y_pred_C, Y_pred_W) = (
#    clust_S.labels_, 
#    clust_A.labels_, 
#    clust_C.labels_, 
#    clust_W.labels_,
#)

#(Y_fitted_S, Y_fitted_A, Y_fitted_C, Y_fitted_W) = (
#    find_perm(3, Y_real, Y_pred_S),
#    find_perm(3, Y_real, Y_pred_A),
#    find_perm(3, Y_real, Y_pred_C),
#    find_perm(3, Y_real, Y_pred_W),
#)

#print(f'fitted single:   {Y_fitted_S}'); print()
#print(f'fitted average:  {Y_fitted_A}'); print()
#print(f'fitted complete: {Y_fitted_C}'); print()
#print(f'fitted ward:     {Y_fitted_W}'); print()

# wybrany do zadania - Warda
#Y_fitted = Y_fitted_W 

#(jac_score_S, jac_score_A, jac_score_C, jac_score_W) = (
#    jaccard_score(Y_real, Y_pred_S, average=None),
#    jaccard_score(Y_real, Y_pred_A, average=None),
#    jaccard_score(Y_real, Y_pred_C, average=None),
#    jaccard_score(Y_real, Y_pred_W, average=None),
#)

#print(f'score single:   {jac_score_S}'); print()
#print(f'score average:  {jac_score_A}'); print()
#print(f'score complete: {jac_score_C}'); print()
#print(f'score ward:     {jac_score_W}'); print()

# 5. Zwizualizuj dane w przestrzeni 2D
#pca = PCA(n_components=2)
#X_reduced = pca.fit_transform(X)

#xplot('2D', X_reduced, Y_real)
#xplot('2D', X_reduced, Y_fitted)
#xplot('2D', X_reduced, Y_real, Y_fitted, classify=True)

# 6. Zwizualizuj dane w przestrzeni 3D

#pca = PCA(n_components=3)
#X_reduced = pca.fit_transform(X)

#xplot('3D', X_reduced, Y_real)
#xplot('3D', X_reduced, Y_fitted)
#xplot('3D', X_reduced, Y_real, Y_fitted, classify=True)


# 7. Narysuj dendrogram
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html

def plot_dendrogram(model, **kwargs):
    # Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([model.children_, 
                                      model.distances_,
                                      counts]).astype(float)
    plt.title('Dendrogram')

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.show()

#model = AgglomerativeClustering(distance_threshold=2, n_clusters=None)
#model = model.fit(X)
## plot_dendrogram(model, truncate_mode='level', p = 3)
#plot_dendrogram(model)

# 8.

# 8.1. K-means 
#k = 3

#clust_K = KMeans(n_clusters=k, random_state=0).fit(X)
#clust_K_fitted = find_perm(k, Y, clust_K.labels_)
#jac_score_K = jaccard_score(Y, clust_K_fitted, average=None)

#print(f'clust_K = { clust_K.labels_} \n')
#print(f'clust_K_fitted = {clust_K_fitted} \n')
#print(f'jaccard_score = {jac_score_K} \n')

### 2D 
#pca = PCA(n_components=2)
#X_reduced = pca.fit_transform(X)

#X_k = KMeans(n_clusters=k, random_state=0).fit(X)
#Y_k_fitted = find_perm(k, Y, X_k.labels_)

#xplot('2D', X_reduced, Y_real)
#xplot('2D', X_reduced, Y_k_fitted)
#xplot('2D', X_reduced, Y_real, Y_k_fitted, classify=True)

#### 3D
#pca = PCA(n_components=3)
#X_reduced = pca.fit_transform(X)

#xplot('3D', X_reduced, Y_real)
#xplot('3D', X_reduced, Y_k_fitted)
#xplot('3D', X_reduced, Y_real, Y_k_fitted, classify=True)

# 8.2. Kmeans lab 6
## Brak

# 8.3. GMM - 343
#k = 3
#clust_gmm = mixture.GaussianMixture(n_components=k).fit(X).predict(X)
#clust_gmm_fitted = find_perm(k, Y, clust_gmm)
#jac_score_gmm = jaccard_score(Y, clust_gmm_fitted, average=None)

#print("clust_gmm = \n", clust_gmm)
#print("clust_gmm_fitted = \n", clust_gmm_fitted)
#print("ac_score_gmm = \n", jac_score_gmm)

### 2D 
#pca = PCA(n_components=2)
#X_reduced = pca.fit_transform(X)

#X_gmm = mixture.GaussianMixture(n_components=k).fit(X)
#Y_gmm_fitted = find_perm(k, Y, X_gmm.predict(X))

#xplot('2D', X_reduced, Y_real)
#xplot('2D', X_reduced, Y_gmm_fitted)
#xplot('2D', X_reduced, Y_real, Y_gmm_fitted, classify=True)

### 3D
#pca = PCA(n_components=3)
#X_reduced = pca.fit_transform(X)

#xplot('3D', X_reduced, Y_real)
#xplot('3D', X_reduced, Y_gmm_fitted)
#xplot('3D', X_reduced, Y_real, Y_gmm_fitted, classify=True)

# 9. 
#csv = pd.read_csv('zoo.csv', sep=";").astype('category').apply(lambda x: x.cat.codes) # Return Series of codes as well as the index.
#X = csv.values[:, 1:-1]
#Y = csv.values[:, -1]

#print(X)
#print(Y)

#k = len(np.unique(Y)) # !!!

# Ward
#clust_ward = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X)
#clust_ward_fitted = find_perm(k, Y, clust_ward.labels_)
#jac_score_ward = jaccard_score(Y, clust_ward_fitted, average=None)

#print("clust_zoo_ward = \n", clust_ward)
#print("clust_zoo_ward_fitted = \n", clust_ward_fitted)
#print("jac_score_ward_zoo = \n", jac_score_ward)

## 2D
#pca = PCA(n_components=2)
#X_reduced = pca.fit_transform(X)

#X_w = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X)
#Y_w_fitted = find_perm(k, Y, X_w.labels_)

#xplot('2D', X_reduced, Y_real, title='oryginał')
#xplot('2D', X_reduced, Y_fitted, title='Ward - rezultat klasteryzacji')
#xplot('2D', X_reduced, Y_real, Y_w_fitted, classify=True, title='różnice')

## 3D
#pca = PCA(n_components=3)
#X_reduced = pca.fit_transform(X)

#X_w = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X)
#Y_w_fitted = find_perm(k, Y, X_w.labels_)

#xplot('3D', X_reduced, Y_real, title='oryginał')
#xplot('3D', X_reduced, Y_w_fitted, title='Ward - rezultat klasteryzacji')
#xplot('3D', X_reduced, Y_real, Y_w_fitted, classify=True, title='różnice')

# Kmeans
#clust_K = KMeans(n_clusters=k, random_state=0).fit(X)
#clust_K_fitted = find_perm(k, Y, clust_K.labels_)
#jac_score_K = jaccard_score(Y, clust_K_fitted, average=None)

#print("clust_K = \n", clust_K)
#print("clust_K_fitted = \n", clust_K_fitted)
#print("jac_score_K = \n", jac_score_K)

### 2D
#pca = PCA(n_components=2)
#X_reduced = pca.fit_transform(X)

#X_k = KMeans(n_clusters=k, random_state=0).fit(X)
#Y_k_fitted = find_perm(k, Y, X_k.labels_)

#xplot('2D', X_reduced, Y_real, title='oryginał')
#xplot('2D', X_reduced, Y_k_fitted, title='Kmeans - rezultat klasteryzacji')
#xplot('2D', X_reduced, Y_real, Y_k_fitted, classify=True, title='różnice')

### 3D
#pca = PCA(n_components=3)
#X_reduced = pca.fit_transform(X)

#X_k =  KMeans(n_clusters=k, random_state=0).fit(X)
#Y_k_fitted = find_perm(k, Y, X_k.labels_)

#xplot('3D', X_reduced, Y_real, title='oryginał')
#xplot('3D', X_reduced, Y_k_fitted, title='Kmeans - rezultat klasteryzacji')
#xplot('3D', X_reduced, Y_real, Y_k_fitted, classify=True, title='różnice')

# Kmeans - lab 6

## GMM
#clust_gmm = mixture.GaussianMixture(n_components=k).fit(X)
#clust_gmm_fitted = find_perm(k, Y, clust_gmm.predict(X))
#jac_score_gmm = jaccard_score(Y, clust_gmm_fitted, average=None)

#print("clust_gmm = \n", clust_gmm)
#print("clust_gmm_fitted = \n", clust_gmm_fitted)
#print("jac_score_gmm = \n", jac_score_gmm)

## 2D
#k = 3
#pca = PCA(n_components=2)
#X_reduced = pca.fit_transform(X)

#X_gmm = mixture.GaussianMixture(n_components=k).fit(X)
#Y_gmm_fitted = find_perm(k, Y, X_gmm.predict(X))

#xplot('2D', X_reduced, Y_real, title='oryginał')
#xplot('2D', X_reduced, Y_gmm_fitted, title='GMM - rezultat klasteryzacji')
#xplot('2D', X_reduced, Y_real, Y_gmm_fitted, classify=True, title='różnice')

### 3D
#pca = PCA(n_components=3)
#X_reduced = pca.fit_transform(X)

#X_gmm = mixture.GaussianMixture(n_components=k).fit(X)
#Y_gmm_fitted = find_perm(k, Y, X_gmm.predict(X))

#xplot('3D', X_reduced, Y_real, title='oryginał')
#xplot('3D', X_reduced, Y_gmm_fitted, title='GMM - rezultat klasteryzacji')
#xplot('3D', X_reduced, Y_real, Y_gmm_fitted, classify=True, title='różnice')


### Kwantyzacja ###
def vectorize(img):
    red = np.array(img[:,:,0]).flatten()
    green = np.array(img[:,:,1]).flatten()
    blue = np.array(img[:,:,2]).flatten()

    red_f = np.array(red)[np.newaxis] # 4 dim

    green_f = np.array(green)[np.newaxis]

    blue_f = np.array(blue)[np.newaxis]

    return np.hstack((red_f.T, green_f.T, blue_f.T))


def quantization(v_img, centers, labels):
    img_quant = v_img.copy()

    return [ centers[labels[i]] for i in range(len(img_quant))]


from sklearn.metrics import mean_squared_error as mean_error

def quantization_error(img, img_quant):
    return [ [ mean_error(img[i,j], img_quant[i,j]) for j in range(img.shape[1])] for i in range(img.shape[0])]


def error_plot(error_img, title):
    plt.imshow(error_img)
    plt.title(title)
    plt.show()


def img_plot(images, title, subtitles):
    (rows, cols) = (1, 2)
    fig, axs = plt.subplots(rows, cols, figsize=(8,4))
    
    for i in range(2):
        img = axs[i].imshow(images[i])
        axs[i].set_title(subtitles[i])
        # fig.colorbar(img, ax=axs[i])

    fig.suptitle(title)
    plt.show()

# 1. 

img = mpimg.imread("image.png")[:,:,:3] # no alpha

print(img.shape)

#plt.imshow(img)
#plt.show()

# 2.
v_img = vectorize(img)

#print(v_img.shape[0])
#print(v_img.shape[1])

# 3. 4. 5. 6. 7.
ks = [5] #, 10] #, 30, 100]

# KMeans

img_quants = []

for k in ks:
    clust_K = KMeans(n_clusters=k, random_state=0).fit(v_img)
    clust_K_centers = clust_K.cluster_centers_
    labels = clust_K.labels_

    print(f'KMeans, k = {k}')
    print("Centers = ", clust_K_centers)
    print("labels = ", labels)

    img_quants.append(clust_K_centers[labels].reshape(img.shape))

for i, k in enumerate(ks):
    img_plot(images=[img, img_quants[i]], title=f'GMM: k = {k}', subtitles=["przed", "po"])
    # error_plot(quantization_error(img, img_quants[i]), title=f'Blad kwantyzacji\nGMM, k = {k}')

# GMM

img_quants = []

for k in ks:
    clust_gmm = mixture.GaussianMixture(n_components=k).fit(v_img)
    clust_gmm_centers = clust_gmm.means_
    labels = clust_gmm.fit_predict(v_img)

    print(f'GMM, k = {k}')
    print("Centers = ", clust_gmm_centers)
    print("labels= ", labels)

    img_quants.append(clust_gmm_centers[labels].reshape(img.shape))
    
for i, k in enumerate(ks):
    img_plot(images=[img, img_quants[i]], title=f'GMM: k = {k}', subtitles=["przed", "po"])

# AgglomerativeClustering nie działał.
# MemoryError: Unable to allocate 352. GiB for an array with shape (47185766400,) and data type float64.

# Brak 8. 9.