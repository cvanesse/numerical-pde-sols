# Generates the meshed for both problems in the project
import distmesh as dm
import numpy as np
from matplotlib import pyplot as plt

# Problem 1 Mesh Generation
h0_rod1 = 0.04
h0_rod2 = 0.035
h0_rect = 0.05
xi = -1.5; xf = 1.5
yi = -1.0; yf = 1.0
x_range = [xi, xf]
y_range = [yi, yf]
bbox = [xi-0.1, yi-0.1, xf+0.1, yf+0.1]

# Mesh inside the first circle
rod_dist = 0.35
fd_rod1 = lambda p: dm.dcircle(p, -rod_dist/2, 0, 0.1)
p_r1, t_r1 = dm.distmesh2d(fd_rod1, dm.huniform, h0_rod1, bbox, pfix=None)

plt.scatter(p_r1[:, 0], p_r1[:, 1], s=5,color="black", marker=".")
plt.title("Rod1 mesh - Problem #1")
plt.show()

fd_rod2 = lambda p: dm.dcircle(p, rod_dist/2, 0, 0.05)
p_r2, t_r2 = dm.distmesh2d(fd_rod2, dm.huniform, h0_rod2, bbox, pfix=None)

plt.scatter(p_r2[:, 0], p_r2[:, 1], s=5, color="black", marker=".")
plt.title("Rod2 mesh - Problem #1")
plt.show()

fd_rect = lambda p: dm.drectangle0(p, xi, xf, yi, yf)

# Full rectangle mesh
pfix = np.vstack((p_r1, p_r2))
p, t = dm.distmesh2d(fd_rect, dm.huniform, h0_rect, bbox, pfix=pfix)

plt.scatter(p[:, 0], p[:, 1], s=10, color="black", marker=".", label="Meshed")
plt.scatter(pfix[:, 0], pfix[:, 1], s=10, color="red", marker=".", label="Fixed")
plt.title("Full mesh - Problem #1")
plt.legend()
plt.show()

# Separate the mesh into separate sections (for easier indexing later)
N_r1 = np.shape(p_r1)[0]
N_r2 = np.shape(p_r2)[0]
N = np.shape(p)[0]
p_sep = np.zeros((N-N_r1-N_r2, 2))

i_sep = 0
for i in range(N):
    point = p[i, :]
    if point[0] in p_r1[:, 0] or point[1] in p_r1[:, 1]:
        continue
    elif point[0] in p_r2[:, 0] or point[1] in p_r2[:, 1]:
        continue
    else:
        p_sep[i_sep, :] = point
        i_sep += 1

plt.scatter(p_sep[:, 0], p_sep[:, 1], s=10, color="green", marker=".")
plt.title("Separated mesh - Problem #1")
plt.show()

# Save the mesh for use later
from os import path
np.savez(path.join(path.dirname(__file__), "meshes/P1_mesh"),
         P=p,
         P_sep=p_sep,
         P_r1=p_r1,
         P_r2=p_r2)
