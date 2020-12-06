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
from finite_difference_methods import cd_1d_matrix_ND_v2

# Load the data from the provided .mat file
mesh_data = loadmat("data/P_array.mat")
P = np.array(mesh_data['P']) # The node coordinates in the computational domain
P = P*1e-6

V_applied = -1e3 # Applied voltage is -1kV

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

K = K / (4*pi*eps_0)

b = np.ones(N)*V_applied

# Solve the integral equation
rho = linalg.solve(K, b)

#print(rho)

print("Plotting the charge distribution...")
triang = triangle_mod.Triangulation(P[:, 0], P[:, 1], triangles=T.simplices)

fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
tpc = ax1.tripcolor(triang, rho, shading='flat')
fig1.colorbar(tpc)
ax1.set_title('Charge concentration on metal tip')
plt.show()

V_in = K.dot(rho)
print("Plotting the Potential distribution...")
fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
tpc = ax1.tripcolor(triang, V_in, shading='flat')
fig1.colorbar(tpc)
ax1.set_title('Potential Distribution on metal tip')
plt.show()

### Compute the potential profile and electric field near the tip of the device


xrange = [1.5e-6, 2.5e-6]
Nx = int((max(xrange)-min(xrange))/dx)
x = np.linspace(min(xrange), max(xrange), Nx)
yrange = [-0.5e-6, 0.5e-6]
Ny = int((max(yrange)-min(yrange))/dy)
y = np.linspace(min(yrange), max(yrange), Ny)

N_out = Nx*Ny
P_out = np.zeros((N_out, 2))

start = 0
end = Ny
for xid in range(Nx):
    P_out[start:end, 0] = x[xid]
    P_out[start:end, 1] = y[:]

    start += Ny
    end += Ny

# Construct integral operator to find the potential in the new domain
# Based on charge from the old domain

def dist(a, b):
    return np.sqrt(np.sum(np.square(a-b)))

eps = 1e-200
K_int = np.zeros((N_out, N))
for i_n in range(N_out):
    r_n = P_out[i_n, :]
    for i_i in range(N):
        r_i = P[i_i, :]
        dni = dist(r_n, r_i)
        if dni >= delta-eps: # If these nodes aren't coincident
            K_int[i_n, i_i] = (dx*dy)/dni
        else:
            K_int[i_n, i_i] = diags

K_int = K_int / (4*pi*eps_0)

V_out = K_int.dot(rho)

print(V_out)
print(P_out)

print("Plotting the potential at the tip...")
T_out = Delaunay(P_out)
triang_out = triangle_mod.Triangulation(P_out[:, 0], P_out[:, 1], triangles=T_out.simplices)

fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
tpc = ax1.tripcolor(triang_out, V_out, shading='flat')
fig1.colorbar(tpc)
ax1.set_title('Potential distribution at the tip')
plt.show()

print("Plotting the electric field near the tip")

# Construct the magnitude of the electric field using central difference
domain = {
    "h": [dy, dx],
    "shape": [Ny, Nx],
    "size": Ny*Nx
}
Dy = cd_1d_matrix_ND_v2(1, 0, domain)
Dx = cd_1d_matrix_ND_v2(1, 1, domain)
Ey = Dy.dot(V_out)
Ex = Dx.dot(V_out)
magE = np.sqrt(np.square(Ey) + np.square(Ex))

fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
tpc = ax1.tripcolor(triang_out, magE, shading='flat')
fig1.colorbar(tpc)
ax1.set_title('Electric Field Magnitude at the tip')
plt.show()


