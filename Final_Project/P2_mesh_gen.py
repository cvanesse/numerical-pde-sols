# Generates the meshed for both problems in the project
import distmesh as dm
import numpy as np
from matplotlib import pyplot as plt

# Problem 2 Mesh Generation
h0_rect = 0.2
h0_cell = h0_rect * (5/8)
h0_elec = h0_cell
xi = -5; xf = 5
yi = 0; yf = 6
x_elec1 = [-3, -1]
x_elec2 = [1, 3]
x_range = [xi, xf]
y_range = [yi, yf]
bbox = [xi-0.1, yi-0.1, xf+0.1, yf+0.1]

# Mesh inside the cell
rod_dist = 0.35
fd_rod1 = lambda p: dm.dcircle(p, 0, 2, 1)
p_r1, t_r1 = dm.distmesh2d(fd_rod1, dm.huniform, h0_cell, bbox)

plt.scatter(p_r1[:, 0], p_r1[:, 1], s=1, marker=".")
plt.title("Cell mesh - Problem #2")
plt.show()

# Rectangle mesh
fd_rect = lambda p: dm.drectangle0(p, xi, xf, yi, yf)

# Add denser mesh around the electrodes
N_elec = int((max(x_elec1) - min(x_elec1)) / h0_elec)
p_e1 = np.zeros((N_elec, 2))
p_e1[:, 0] = np.linspace(min(x_elec1), max(x_elec1), N_elec)

p_e2 = p_e1.copy()
p_e2[:, 0] = np.linspace(min(x_elec2), max(x_elec2), N_elec)

pfix = np.vstack((p_r1, p_e1, p_e2))

# Full rectangle mesh
p, t = dm.distmesh2d(fd_rect, dm.huniform, h0_rect, bbox, pfix=pfix)

plt.scatter(p[:, 0], p[:, 1], s=10, color="black", marker=".", label="Meshed")
plt.scatter(pfix[:, 0], pfix[:, 1], s=10, color="red", marker=".", label="Fixed")
plt.title("Full mesh - Problem #2")
plt.legend()
plt.show()

# Separate the mesh into separate sections (for easier indexing later)
N_r1 = np.shape(p_r1)[0]
N_e1 = np.shape(p_e1)[0]
N_e2 = np.shape(p_e2)[0]
N = np.shape(p)[0]
p_sep = np.zeros((N-N_r1-N_e2-N_e1, 2))

i_sep = 0
for i in range(N):
    point = p[i, :]
    if point[0] in p_r1[:, 0] or point[1] in p_r1[:, 1]:
        continue
    elif point[0] in p_e1[:, 0] or point[1] in p_e1[:, 1]:
        continue
    elif point[0] in p_e2[:, 0] or point[1] in p_e2[:, 1]:
        continue
    else:
        p_sep[i_sep, :] = point
        i_sep += 1

plt.scatter(p_sep[:, 0], p_sep[:, 1], s=10, color="green", marker=".")
plt.title("Separated mesh - Problem #2")
plt.show()

# Save the mesh for use later
from os import path
np.savez(path.join(path.dirname(__file__), "meshes/P2_mesh"),
         P=p, T=t, P_sep=p_sep,
         P_e1=p_e1, P_e2=p_e2,
         P_c=p_r1)
