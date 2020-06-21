import numpy as np
import math

from numpy.lib.stride_tricks import as_strided

print("Zadania wstępne.")

print(); print()

### 1. Tablice ###
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1,2,3,4,5], [6,7,8,9,10]])
bt = np.transpose(b)

print(np.arange(0, 100)) # <0, 100)

print(np.linspace(0, 2, 10))

np.arange(0, 100, 5)

print(); print()

### 2. Liczby losowe ###

np.around(np.random.randn(1, 20), 2)

np.random.randint(1, 1000, 100)

np.zeros((3, 2))
np.ones((3, 2))

matrix = np.matrix(np.random.randint(0, 100, 25).reshape(5, 5), int)

print(matrix)

## Zadanie
a = np.random.uniform(0, 10, 10)

b = np.array(a, int)
print(b)

a = np.around(a)
a = a.astype(int)
print(a)

print(); print()

### 3. Selekcja danych ###
b = np.array([[1,2,3,4,5], [6,7,8,9,10]],int)
print(b.ndim) # wymiary
print(b.size) # liczba elementów

print(b[0, 1]) # 2
print(b[0, 3]) # 4
print(b[0])
print(b[:, 1])

dim = (20, 7)
print(np.random.randint(0, 100, dim[0]*dim[1]).reshape(dim)[:, 0:4])

print(); print()

# 4. Operacje matematyczne i logiczne
a = np.random.randint(0, 10, 9).reshape(3, 3)
b = np.random.randint(0, 10, 9).reshape(3, 3)

print("a: ")
print(a)

print("b: ")
print(b)

print("Sum: ")
print(np.add(a, b))

print("Mul: ")
print(np.multiply(a, b))

print("Divide: ")
print(np.divide(a, b))

print("Power: ")
print(np.power(a, b))

print(a >= 4)

# Sprawdź czy wartość macierzy a 1¿= ¡=4. ???

print("Trace b: ", end='')
print(np.trace(b))

print(); print()

### 5. Dane statystyczne ###
print("Sum b: ", end='')
print(np.sum(b))

print("Min b: ", end='')
print(b.min())

print("Max b: ", end='')
print(b.max())

print("STD b: ", end='')
print(np.std(b))

print("avg row b: ")
print(np.average(b, 1))

print("avg col b: ")
print(np.average(b, 0))

print(); print()

### 6. Rzutowanie wymiarów za pomocą shape lub resize ###

x = np.array(np.random.uniform(0, 100, 50), int)
print(x)

X = np.reshape(x, (10, 5))
print(X)

X = np.resize(x, (10, 5))
print(X)

# np.ravel - jak [...array]

a = np.random.uniform(0, 10, 5)
b = np.random.uniform(0, 10, 4)

print(a)
print(b)

n_a = a[:, np.newaxis]
print(n_a)
print(n_a + b)

print(); print()

### 7. Sortowanie danych ###

a = np.random.randn(5, 5)

print(a)

# wierszowo od najmniejszej
b = np.sort(a)

print(b)

# kolumnowo od największej
c = np.transpose(a)
c = np.sort(c)
c = np.transpose(c)[::-1]

print(c)

b = np.array([
    (1, 'MZ', 'mazowieckie'),
    (2, 'ZP', 'zachodniopomorskie'),
    (3, 'ML', 'małopolskie')
])

b = b.reshape(3, 3)

print(b)

b = b[b[:,2].argsort()]

print(b[2, 2])

print(); print()

##### Zadania podsumowujące #####

print("Zadania podsumowujące.")

print()

# 1
print("1: ")
a = np.random.randint(0, 10, 50).reshape(10, 5)

print(a)
print(np.trace(a))
print(np.diag(a))

print(); print()

# 2
print("2: ")
a = np.random.randn(5)
b = np.random.randn(5)

print(a)
print(b)
print(np.multiply(a, b))

print(); print()

# 3
print("3: ")
a = np.random.randint(1, 100, 25)
b = np.random.randint(1, 100, 25)

print(a)
print(b)

a = np.reshape(a, (5, 5))
b = np.reshape(b, (5, 5))

print(np.add(a, b))

print(); print()

# 4
print("4: ")
a = np.random.randint(0, 20, 20).reshape(5, 4)
b = np.random.randint(0, 20, 20).reshape(4, 5)

a = np.reshape(a,(4, 5))

print(a)
print(b)
print(np.add(a, b))

print(); print()

# 5
print("5: ")
print(np.multiply(a[:, 3], a[:, 4]))
print(np.multiply(b[:, 3], b[:, 4]))

print(); print()

# 6
print("6: ")
a = np.random.normal(0, 0.1, 9).reshape(3, 3)
b = np.random.uniform(0, 1, 9).reshape(3, 3)


print(a)
print(b)

print(np.mean(a))
print(np.mean(b))

print(np.std(a))
print(np.std(b))

print(math.sqrt(np.std(a)))
print(math.sqrt(np.std(b)))

print(); print()

# 7
print("7: ")
dim = (10, 10)
a = np.random.randint(1, 10, dim[0]*dim[1]).reshape(dim)
b = np.random.randint(1, 10, dim[0]*dim[1]).reshape(dim)

print(a*b)
print(np.dot(a, b))

print(); print()

# 8
print("8: ")
print(a)
print(as_strided(a.T, (3, 5), a.strides))

print(); print()

# 9
print("9: ")
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
print(np.vstack((a, b)))
print(np.hstack((a, b)))

print(); print()

# 10
print("10: ")
matrix = np.arange(24).reshape(4, 6)

maxtrix = np.array([as_strided(matrix[0], (2, 3), matrix.strides).max(),
                    as_strided(matrix[0][3:6], (2, 3), matrix.strides).max(),
                    as_strided(matrix[2], (2, 3), matrix.strides).max(),
                    as_strided(matrix[2][3:6], (2, 3), matrix.strides).max()
                    ]).reshape(2, 2)

print(matrix)
print(maxtrix)

print(); print()
