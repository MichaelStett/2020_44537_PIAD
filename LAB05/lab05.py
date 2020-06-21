import numpy as np
import pandas as pd

from sklearn.datasets import fetch_rcv1
from scipy import sparse


# 1.
def freq(x, prob=True):
    counts = pd.value_counts(x, normalize=prob)
    return [counts.keys().tolist(), counts.tolist()]

# 2.
def freq2(x, y, prob=True):
    df = pd.DataFrame({
        'x': x,
        'y': y
    })

    df = df.groupby(["x", "y"]).size().reset_index(name='n')

    xi = df["x"].tolist()
    yi = df["y"].tolist()
    ni = df["n"].tolist() # name='n'

    pi = [ (float(n) / sum(ni)) for n in ni]

    if prob == True:
        return [xi, yi, pi]
    else:
        return [xi, yi, ni]

# 3.
def entropy(x): # H  = -E( p * log2(p))
    [xi, pi] = freq(x, prob=True)

    return -sum([ (p * np.log2(p)) for p in pi])

def conditional_entropy(y, x):
    [yi, xi, pi] = freq2(y, x, prob=True)

    return sum([ (P * entropy([y for y, x in zip(yi, xi) if x == X])) for X, P in zip(xi, pi)])

def infogain(x, y): # I(Y, X) = H(Y) - H(Y|X)
    return entropy(y) - conditional_entropy(y, x)

data = pd.read_csv("zoo.csv")

#print(data)

columns = data.columns[:-1]

gains = [ infogain(data[i], data["type"]) for i in columns]

new_data = pd.DataFrame({ 
    'kolumna': columns, 
    'przyrost': gains
})

print("Przyrost informacji: ")
print(new_data.sort_values(by=['przyrost'], ascending=False))
