import numpy as np
import matplotlib.pyplot as plt

x = list(range(1, 101))
# create a decreasing list of random numbers from 8000 to 4000
y = sorted(np.random.randint(4000, 8000, 100), reverse=True)

plt.plot(x, y)
plt.axis([-5, 110, 0, 9000])
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

x = list(range(1, 101))
# create a decreasing list of random numbers from 8000 to 4000
y = sorted(np.random.randint(9, 13, 100), reverse=True)

plt.plot(x, y)
plt.axis([-5, 110, 0, 100])
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.show()
