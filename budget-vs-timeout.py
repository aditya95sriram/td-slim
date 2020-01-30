import numpy as np
import matplotlib.cm as colormap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from csv import DictReader
import sys

budgets = [5,10,15,20,25,30]
timeouts = [2,10,20,100,200]

X = np.array([budgets] * len(timeouts))
Y = np.array([timeouts] * len(budgets)).T

data = dict()

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "heur_015.gr"

numverts = numedges = None
runtime = 0

best_best_depth = 1e9
with open("../budget-vs-timeout/budget-vs-timeout.csv") as csvfile:
    reader = DictReader(csvfile)
    for row in reader:
        if row['filename'] != filename: continue
        if numverts is None:
            numverts = int(row['n'])
            numedges = int(row['m'])
        runtime += float(row['time'])
        budget = int(row['budget'])
        timeout = int(row['timeout'])
        best_depth = int(row['best_depth']) if row['best_depth'] else 1e9
        best_best_depth = min(best_best_depth, best_depth)
        data[(budget, timeout)] = best_depth


Z = np.array([[data[(xi, yi)] / best_best_depth for xi in budgets] for yi in timeouts])

print(X.shape, Y.shape, Z.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(X, Y, Z)
cset = ax.contour(X, Y, Z, zdir='x', offset=0, colors="black")
cset = ax.contour(X, Y, Z, zdir='y', offset=250, colors="black")

ax.set_xlim3d(0, 35)
ax.set_ylim3d(0, 250)
ax.set_title(f"{filename} (n={numverts}, m={numedges}, t={runtime:.3f}s)")
# ax.set_zlim3d(1, 2)


ax.set_xlabel("budget")
ax.set_ylabel("timeout")
ax.set_zlabel("depth")
plt.show()