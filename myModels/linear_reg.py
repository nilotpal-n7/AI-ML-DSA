import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

n = 1000
r = 1000
data = pd.DataFrame([[j, 1.34 * j + 2.34 + random.random() * random.random() * r] for j in range(n)], columns=["x", "y"], index=[i for i in range(n)])

def loss_fun(m, b, data):
    total_error = 0
    n = len(data)

    for i in range(n):
        total_error += (data.iloc[i].y - (m * data.iloc[i].x + b)) ** 2
    total_error /= n

    return total_error

def gradient_des(data, m, b, L):
    n = len(data)
    diff_m = 0
    diff_b = 0

    for i in range(n):
        y = data.iloc[i].y
        x = data.iloc[i].x
        t = y - (m * x + b)

        diff_m += (-2 / n) * x * t
        diff_b += (-2 / n) * t

    m = m - L * diff_m
    b = b - L * diff_b
    return m, b

m = 0
b = 0
L = pow(10, -6)
epochs = 1001

for i in range(epochs):
    if(i % 10 == 0):
        print(i)
    m, b = gradient_des(data, m, b, L)
print(m, b)

plt.scatter(data.x, data.y, color="black")
plt.plot(data.x, m * data.x + b, color="red")
plt.show()
