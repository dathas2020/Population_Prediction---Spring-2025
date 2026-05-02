import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Data.csv")

from sklearn.linear_model import LinearRegression


D1 = df.loc[df["District"] == "Darjiling"]
D2 = df.loc[df["District"] == "Medinipur"]

x = D1.Year
y = D1.Total

print(x)
print(y)

X = np.array([x])
Y = np.array([y])

print(X)
print(Y)

x_pred = np.array([2031, 2041, 2051])

model = LinearRegression()
model.fit(X.reshape(1, -1), Y)
y_pred = model.predict(x_pred.reshape(-1, 1))
print("y_pred:", y_pred)

library = ['Male', 'Female', 'Total']

enthusiasts_north = [10480,28120,38600]
enthusiasts_south = [37617,50289,87906]
bar_width = 0.35
x = np.arange(len(library))

plt.bar(x - bar_width/2, enthusiasts_north, bar_width, label='Koch Bihar', color='skyblue')
plt.bar(x + bar_width/2, enthusiasts_south, bar_width, label='Jalpaiguri', color='lightcoral')

plt.xlabel('Population Categories')
plt.ylabel('Population')
plt.title('Comparision of Koch Bihar and Jalpaiguri 1991')
plt.xticks(x, library)
plt.legend(title='Districts')
plt.show()

