import numpy as np

x = np.random.randint(0, 5, size=(2,2,2))
print(x)
x = np.where(x > 2, 1, 0)
print(x)

