import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import style

style.use('ggplot')

COLORS = [
    'r', 'g', 'b', 'c', 'm', 'y'
]


def distp(X, C):
    temp = np.zeros(shape=(len(X), len(C)))

    for i, xi in enumerate(X):
        for j, cj in enumerate(C):
            sub = np.subtract(xi, cj)
            temp[i, j] = np.sqrt(np.dot(sub, sub.T))

    return temp


def distm(X, C, V):
    temp = np.zeros(shape=(len(X), len(C)))

    for i, xi in enumerate(X):
        for j, cj in enumerate(C):
            sub = np.subtract(xi, cj)
            temp[i, j] = np.sqrt(np.dot(np.dot(sub, np.linalg.inv(V)), sub.T))
    
    return temp


def ksrodki(X, K=2):
    (n, m) = (X.shape[0], X.shape[1])

    P = np.zeros(shape=(n, K))

    # 1. Wektory macierzy środków C są inicjowane losowo.
    C = X[np.random.randint(n, size=K), :]

    while True:
        (prev_C, prev_P)  = (C, P)

        # 2. Minimalna odległość
        P = np.zeros(shape=(n, K))

        D = distp(X, C) 

        for i in range(n):
            P[i, np.argmin(D[i])] = 1

        # 3. Obliczyć C[k]
        for k in range(K):
            denominator = sum(P[i, k] for i in range(n))
            
            if denominator == True:
                C[k] = sum(P[i, k] * X[i] for i in range(n)) / denominator

        # 4. Powtarzać kroki 2 i 3 dopóki grupowanie nie ustabilizuje się
        eps = 10**-5

        if not (np.max(P - prev_P) >= eps) and not(np.max(C - prev_C) >= eps):
            break

    # 5. Każdy obiekt xi należy do klasy k w przypadku, gdy P[i, k] = 1.
    CX = [ [] for k in range(K) ]

    for i in range(n):
        for k in range(K):
            if P[i, k] == 1:
                CX[k].append(X[i])

    for i in range(k):
        C[i] = np.mean(CX[i], axis=0)

    return C, CX


def F_dividend(C, CX):
    dividend = 0
    n = 0

    cl = np.mean(np.concatenate(CX, axis=0), axis=0)

    for c in C:
        dividend += distp(np.asmatrix(c), np.asmatrix(cl))
        n += 1

    dividend /= n

    return dividend


def F_divisor(C, CX):
    divisor = 0
    n = 0

    for i, c in enumerate(C):
        for cx in CX[i]: # Ck
            divisor += distp(np.asmatrix(cx), np.asmatrix(c)) ** 2
            n += 1

    divisor /= n

    return divisor


def F(C, CX):
    (dividend, divisor) = (
        F_dividend(C, CX),
        F_divisor(C, CX),
    )

    #print(f'{dividend} / {divisor}')
    return np.ravel(dividend / divisor)[0]


# Zadanie
data = pd.read_csv("autos.csv", sep=";")

X = np.array(data[["width", "height"]]).astype(float)

#plt.plot(X[:, 0], X[:, 1], ".", color='k', markersize=5)
#plt.show()

K = 4

C, CX = ksrodki(X, K)

CX = [ np.array(cx) for cx in CX ]
C  = [ np.array(c) for c in C ]

print(CX)
cetroid_color = list(zip(map(tuple, C), COLORS))

print(pd.DataFrame(cetroid_color, columns=["point", "color"]).to_string(index=False))

print(f'jakość grupowania: {F(C, CX)}')


for i, cx, c in zip(range(K), CX, C):
    color = COLORS[i % len(COLORS)]
    if len(cx) > 0: # not empty
        plt.plot(cx[:, 0], cx[:, 1], ".", color=color, markersize=5)

    plt.plot(c[0], c[1], "o", color=color, markersize=10)

plt.title(f'KMeans: k = {K}')
plt.show()
