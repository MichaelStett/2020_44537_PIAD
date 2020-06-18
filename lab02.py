import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Laboratorium 2 – PANDAS
print("Zadania rozgrzewkowe")
print("1.")
nums = np.random.normal(0,1,size=(5,3))
dateRange = pd.date_range(start='3/1/2020', end='3/5/2020')
dateNumFrame = pd.DataFrame(data=nums, index=dateRange, columns=['A','B','C'])
dateNumFrame.index.name = "data"
print(dateNumFrame)

print("2.")
numFrame = pd.DataFrame(data=np.random.randint(-1000,2000,(20,3)), index=np.arange(20), columns=list("ABC"))
numFrame.index.name = "id"
print(numFrame)

print("2.1:"); print(numFrame.head(3))
print("2.2:"); print(numFrame.tail(3))
print("2.3:"); print(numFrame.index.name)
print("2.4:"); print(list(numFrame.columns))
print("2.5:"); print(numFrame.to_string(index=False, header=False))
print("2.6:"); print(numFrame.sample(5))
print("2.7.1:"); print(numFrame["A"])
print("2.7.2:"); print(numFrame.loc[:, ["A", "B"]]) # wszystkie wiersze z dwóch kolumn
print("2.8.1:"); print(numFrame.iloc[np.arange(3),[0,1]]) # 3 wiersze z kolumny 0 i 1
print("2.8.2:"); print(numFrame.iloc[[5]]) # 5 wiersz
print("2.8.3:"); print(numFrame.iloc[[0,5,6,7],[1,2]])
print("2.9.1."); print(numFrame.describe()) # rozkład ststaystyczny nie noramalny
print("2.9.2:"); print(numFrame > 0)
print("2.9.3:"); print(numFrame[numFrame > 0])
print("2.9.4:"); print(numFrame.loc[:, 'A'][numFrame.loc[:, 'A'] > 0])
print("2.9.5:"); print(numFrame.loc[:, numFrame.columns != "id"].mean(axis = 0))
print("2.9.6:"); print(numFrame.mean(axis = 1))

print("3.")
datesFrame = pd.DataFrame({"data" : dateRange})
nums = np.random.normal(0,1,size=(5,3))
numFrame = pd.DataFrame(nums, columns=list("ABC"))
newFrame = pd.concat([datesFrame, numFrame],axis=1)
print(newFrame)
print(newFrame.transpose())

print("4.")
df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5], 
        "y": ['a', 'b', 'a', 'b', 'b']
        }, index=np.arange(5))
df.index.name='id'
print("4.1:"); print(df.sort_values(by=['id']))
print("4.2:"); print(df.sort_values(by=['y'], ascending=False))

print("5.")
slownik = {
    'Day': ['Mon', 'Tue', 'Mon', 'Tue', 'Mon'],
    'Fruit': ['Apple', 'Apple', 'Banana', 'Banana', 'Apple'],
    'Pound': [10, 15, 50, 40, 5],
    'Profit': [20, 30, 25, 20, 10]
}
df3 = pd.DataFrame(slownik)
print(df3)
print(df3.groupby('Day').sum())
print(df3.groupby(['Day','Fruit']).sum())

print("6.")
df = pd.DataFrame(np.random.randn(20, 3), index=np.arange(20), columns=['A','B','C'])
df.index.name='id'
print(df)

df['B'] = 1 # wszystkie wiersze kolumny 'B' ustawia na 1
print(df)

df.iloc[1,2] = 10 # wartość elementu (1, 2) ustawia na 10
print(df)

df[df < 0]=-df # wartość bezwzględna
print(df)

print("7.")
df.iloc[[0, 3], 1] = np.nan # w kolumnie 1 dla wierszy 0 i 3 ustwaia wartość na NaN
print(df)

df.fillna(0, inplace=True) # w miejsce NaN ustwaia wartosc 0
print(df)

df.iloc[[0, 3], 1] = np.nan
df = df.replace(to_replace=np.nan,value=-9999) # zamienia wartosci NaN na -9999
print(df)

df.iloc[[0, 3], 1] = np.nan
print(pd.isnull(df)) # zwraca informacje o tym czy na danej pozycji występuje NaN

# --------------------- #

# Zadania
print("Zadania:")
df = pd.DataFrame({ "x": [1, 2, 3, 4, 5], 'y': ['a', 'b', 'a', 'b', 'b']})
print("df = "); print(df)

# 1.
print("1.")
print(df.groupby('y').mean())

# 2.
print("2.")
print(df['x'].value_counts())
print(df['y'].value_counts())

# 3.
print("3.")
df1 = np.loadtxt("autos.csv", dtype='str', delimiter=';')
df2 = pd.read_csv("autos.csv", delimiter=';')

print("df1 = "); print(df1)
print("df2 = "); print(df2)
# np.loadtxt() zwraca macierz, w pierwszej kolumnie jest numeracja wierszy, pierwszy wiersz nagłówki, wymaga ustawienia dtype='str'
# pd.read_csv() zwraca DataFrame z wartosciami i naglowkami z pliku
# dodatkowo musiałem ustawić delimeter=';' bo tak miałem rozdzielone wartości w pliku csv

print("4."); print(df2.groupby('make')[['city-mpg','highway-mpg']].mean().mean(axis=1))
print("5."); print(df2.groupby('make')['fuel-type'].value_counts())
print("6.")
(term, var, var2, dotlabel) = ("length", "city-mpg", "width", "próbki")

fit_1 = np.polyfit(df2[term], df2[var], 1) # obliczenie wspł. wielomianu
fit_2 = np.polyfit(df2[term], df2[var], 5)
print(fit_1)
print(fit_2)

print("7."); print(stats.pearsonr(df2[term], df2[var]))

print("8.")
x = np.linspace(df2[term].min(), df2[term].max(), 500)
plt.plot(df2[term], df2[var], 'b.', label=dotlabel)
plt.plot(x, np.polyval(fit_1, x)) # obliczenie wartości wielomianu
plt.plot(x, np.polyval(fit_2, x))
plt.xlabel(term)
plt.ylabel(var)
plt.legend()
plt.show()

print("9.")
y = stats.gaussian_kde(df2[term])
plt.plot(x, y(x), label='Density function')
plt.plot(df2[term], y(df2[term]), 'r.', label=dotlabel)
plt.legend()
plt.show()

print("10.")
ax = plt.subplot(211)
ax.plot(x, y(x), label=f'{term} - function')
ax.plot(df2[term], y(df2[term]), 'r.', label=f'{term} - {dotlabel}')
plt.legend()

x = np.linspace(df2[var2].min(), df2[var2].max(), 500)
y = stats.gaussian_kde(df2[var2])
bx = plt.subplot(212)
bx.plot(x, y(x), label=f'{var2} - function')
bx.plot(df2[var2], y(df2[var2]), 'r.', label=f'{var2} - {dotlabel}')
plt.legend()
plt.show()

print("11.")
xmin = df2[var2].min()
xmax = df2[var2].max()
ymin = df2[term].min()
ymax = df2[term].max()

y = stats.gaussian_kde(np.vstack([df2[var2],df2[term]]))

xv, yv = np.meshgrid(
    np.linspace(xmin, xmax, 100),
    np.linspace(ymin, ymax, 100)
)

z = np.reshape(
    y(np.vstack([
        xv.ravel(),
        yv.ravel()
    ])).T,
    xv.shape
)

fig = plt.figure()
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

plt.plot(df2[var2], df2[term], 'b.')

cntrf = ax.contourf(xv, yv, z, cmap='Reds')
cntr = ax.contour(xv, yv, z, colors='k')

ax.clabel(cntr, inline=1, fontsize=8)
ax.set_xlabel(var2)
ax.set_ylabel(term)
plt.savefig('11.png')
plt.savefig('11.pdf')
plt.show()
