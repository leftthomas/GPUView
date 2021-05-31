import matplotlib.pyplot as plt
import numpy as np

x = np.array([3, 4, 5, 6])
y = np.array([3, 4, 5, 6])
z = np.array([[33.1, 33.6, 33.4, 32.8], [33.7, 36.0, 32.9, 34.1],
              [33.9, 34.1, 32.9, 32.7], [34.7, 33.5, 31.6, 33.0]])

xx, yy = np.meshgrid(x - 0.25, y - 0.25)
x, y = xx.ravel(), yy.ravel()
bottom = np.zeros_like(x)
z = z.ravel()

width = height = 0.7

fig = plt.figure()
ax = fig.gca(projection='3d')
colors = plt.cm.jet((z.argsort() + 1) / 50)
ax.bar3d(x, y, bottom, width, height, z, color=colors, shade=True)
ax.set_xlabel('k')
ax.set_ylabel('m')
ax.set_zlabel('mAP')
plt.savefig('hyper.pdf', bbox_inches='tight', pad_inches=0)
