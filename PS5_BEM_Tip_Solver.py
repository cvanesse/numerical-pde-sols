# Uses the boundary-element method to solve for the charge density on a ti

from scipy.io import loadmat
import numpy as np
from scipy import sparse
import math
from scipy.spatial import Delaunay
from numpy import linalg
from matplotlib import pyplot as plt
from matplotlib import tri as triangle_mod
from bisect import bisect_left

# Load the data from the provided .mat file
mesh_data = loadmat("data/P_array.mat")
P = np.array(mesh_data['P']) # The node coordinates in the computational domain
P = P*1e-6

V_applied = -1e-3 # Applied voltage is -1kV

dx = 0.05e-6
dy = 0.025e-6

#L = 2e-6
#N = 21
#x = np.linspace(0, L, N)
#y = np.linspace(0, L, N)
#P = np.zeros((N*N, 2))

#start = 0
#end = N
#for xid in range(N):
#    P[start:end, 0] = x[xid]
#    P[start:end, 1] = y[:]

#    start += N
#    end += N

#dx = L/N
#dy = L/N
#V_applied = 1

T = Delaunay(P)

# Physical constants
eps_0 = 8.85419e-12
pi = np.pi


# Construct the BEM matrix
# Kij = 1/(4*pi*eps_0) * 1/(Rij)

N = np.shape(P)[0]
K = np.zeros((N, N))

# Fill in the off-diagonals
for i in range(N):
    ri = P[i, :]
    for j in range(N):
        if i != j:
            rj = P[j, :]
            rij = ri - rj
            K[i, j] = (dy*dx)/np.sqrt(np.sum(np.power(rij, 2)))


# Fill in the on-diagonals
K_diags = np.eye(N)
delta = math.sqrt(dx**2 + dy**2)
diags = 2*(dx*math.log((delta+dy)/dx) + dy*math.log((delta+dx)/dy))
K_diags = K_diags * diags

K = K + K_diags

print(K)

K = K / (4*pi*eps_0)

b = np.ones(N)*V_applied

# Solve the integral equation
q = 1.602e-19
rho = linalg.solve(K, b)

#print(rho)

print("Plotting the results...")
triang = triangle_mod.Triangulation(P[:, 0], P[:, 1], triangles=T.simplices)

fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
tpc = ax1.tripcolor(triang, rho, shading='flat')
fig1.colorbar(tpc)
ax1.set_title('Charge concentration on metal tip')
plt.show()

