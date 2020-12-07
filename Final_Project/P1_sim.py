# Runs the simulation for problem 1
import numpy as np
from os import path
from finite_element_poisson import *
from matplotlib import pyplot as plt
from matplotlib import tri as triangle_mod

## Physical information
eps_0 = 8.854187e-12
eps_r = 1
V_r1 = 1
V_r2 = -1

## Script config
show_mesh_plots = True
show_sparsity = True

curdir = path.dirname(__file__)

# Load the mesh, do delaunay triangulation
meshdata = np.load(path.join(curdir, "meshes/P1_mesh.npz"))

# Load all of the mesh information
P = meshdata["P"]
N = np.shape(P)[0]
T = meshdata["T"]
N_e = np.shape(T)[0]
p_fs = meshdata["P_sep"]
p_r1 = meshdata["P_r1"]
p_r2 = meshdata["P_r2"]

x_min = np.min(P[:, 0])
x_max = np.max(P[:, 0])
y_min = np.min(P[:, 1])
y_max = np.max(P[:, 1])
eps_select = 1e-5 # The minimum distance from the boundary for a point to be considered a boundary node.

# Calculate the indices of different sections of the domain
#   (For easier indexing/application of BCs)
n_bdry = []
n_cent = []
n_r1   = []
n_r2   = []
for i in range(N):
    ri = P[i, :]
    if ri[0] in p_r1[:, 0] and ri[1] in p_r1[:, 1]:
        n_r1.append(i)
    elif ri[0] in p_r2[:, 0] and ri[1] in p_r2[:, 1]:
        n_r2.append(i)
    else:
        lb = abs(ri[0] - x_min) < eps_select
        rb = abs(ri[0] - x_max) < eps_select
        bb = abs(ri[1] - y_min) < eps_select
        tb = abs(ri[1] - y_max) < eps_select
        if lb or rb or tb or bb:
            n_bdry.append(i)
        else:
            n_cent.append(i)

n_bdry = np.array(n_bdry)
n_cent = np.array(n_cent)
n_r1 = np.array(n_r1)
n_r2 = np.array(n_r2)

if show_mesh_plots:
    plt.scatter(P[n_bdry, 0], P[n_bdry, 1], s=5, color="red", marker=".", label="Boundary")
    plt.scatter(P[n_r1, 0], P[n_r1, 1], s=5, color="red", marker=".", label="Rod 1")
    plt.scatter(P[n_r2, 0], P[n_r2, 1], s=5, color="red", marker=".", label="Rod 2")
    plt.scatter(P[n_cent, 0], P[n_cent, 1], s=5, color="black", marker=".", label="Internal")
    plt.title("Identified Regions")
    plt.show()

# Remap the domain for simpler indexing of points
#   First nodes are boundary nodes
#   Second set of nodes are R1 nodes
#   Third set of nodes are R2 nodes
#   Fourth set of nodes are internal nodes

# Create the new point order and invert the mapping
new_point_order = np.hstack((n_bdry, n_r1, n_r2, n_cent))
invmap = np.zeros_like(new_point_order)
for j in range(N): invmap[new_point_order[j]] = j

T_new = np.zeros_like(T)
for e in range(len(T_new)):
    T_new[e, :] = invmap[T[e, :]]
P_new = P[new_point_order, :]

if show_mesh_plots:
    plt.scatter(P[:, 0], P[:, 1], s=5, c=np.linspace(0, N, N))
    plt.title("Original Point Ordering")
    plt.show()
    plt.scatter(P_new[:, 0], P_new[:, 1], s=5, c=np.linspace(0, N, N))
    plt.title("Updated Point Ordering")
    plt.show()

# Update with the new point ordering
n_bdry = invmap[n_bdry]
n_r1 = invmap[n_r1]
n_r2 = invmap[n_r2]
n_cent = invmap[n_cent]
P = P_new
T = T_new

# Calculate the circumcenters and radii for the entire domain
C = calculate_circumcenters(P, T)
if show_mesh_plots:
    plt.scatter(C[:, 0], C[:, 1], s=5, color="red", marker=".", label="Circumcenters")
    plt.scatter(P[:, 0], P[:, 1], s=5, color="black", marker=".", label="Points")
    plt.title("Points with circumcenters")
    plt.show()

## At this point, the entire domain has been sorted for simpler indexing, and all circumcenters are calculated

# Calculate the stiffness matrix
eps_e = eps_r*eps_0*np.ones((N_e)) # The permittivity at the center of each triangle
eps_p = eps_r*eps_0*np.ones((N))   # The permittivity at each point

print("Constructing Stiffness Matrix...")
K, b = construct_poisson_eq(P, T, eps_e, eps_p)

# Verify validity of the matrix
valid = True
eps_test = 1e-30
for i in range(N):
    valid = valid and np.abs(np.sum(K[i, :])) < eps_test

if not valid:
    print("ERROR: Invalid Stiffness Matrix - Sum of rows are not all 0.")

if show_sparsity:
    plt.spy(K, markersize=0.5)
    plt.title("Sparsity of Stiffness Matrix Before BC Application")
    plt.show()

# Apply the dirichlet conditions (rod voltages) to the FE equation
V_rods = np.zeros(np.size(n_r1) + np.size(n_r2))
V_rods[:np.size(n_r1)]  = V_r1
V_rods[-np.size(n_r2):] = V_r2
n_rods = np.hstack((n_r1, n_r2))
n_rest = np.hstack((n_cent, n_bdry))
K, b = apply_dirichlet_conditions(K, b, np.hstack((n_r1, n_r2)), n_rest, V_rods)

#print(extract_submatrix(K, n_rods, n_rods))
#print(K[n_rods, n_rods])
#print(extract_submatrix(K, n_rods, n_rest))
#print(extract_submatrix(K, n_rest, n_rods))

if show_sparsity:
    plt.spy(K, markersize=0.5)
    plt.title("Sparsity of Final Stiffness Matrix")
    plt.show()

K = K.tocsr()
V = linalg.spsolve(K, b)

print("Plotting the potential distribution...")
triang_out = triangle_mod.Triangulation(P[:, 0], P[:, 1], triangles=T)

fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
tpc = ax1.tripcolor(triang_out, V, shading='flat')
fig1.colorbar(tpc)
ax1.set_title('Potential distribution')
plt.show()
