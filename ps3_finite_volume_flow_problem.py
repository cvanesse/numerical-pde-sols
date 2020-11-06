# Finite-Volume solver for the flow problem
from scipy.io import loadmat
import numpy as np
from scipy import sparse
import math
from scipy.sparse import linalg
from matplotlib import pyplot as plt
from matplotlib import tri as triangle_mod
from bisect import bisect_left

## Load simulation grid from provided .mat file
mesh_data = loadmat("data/channel_mesh.mat")
P = np.array(mesh_data['P']) # The node coordinates in the computational domain
T = np.array(mesh_data['T'])-1 # The node indexes of each triangle (converted to python indexing)

# Determine the number of nodes in the simulation domain
Npts = np.shape(P)[0]
Ntri = np.shape(T)[0]

## Simulation Parameters
sim_name = "Open Top"
use_top = False # Set to True to apply the no-slip condition at the top of the channel

pz = 1 # Pressure gradient in z (transport) direction
R  = 1 # Reynold's number

# Initialize variables for the discrete equation
L = sparse.lil_matrix((Npts, Npts))
A = np.zeros((Npts, 1))

## Van's algorithm

# Finds the circumcenter of the triangle defined by coordinates of the rows of the triangle
def find_circumcenter(tri):
    zeros = np.zeros((3, 1))
    tri = np.concatenate((tri, zeros), axis=1)
    ri = tri[0, :]

    dr = [tri[i+1, :]-ri for i in range(2)]
    mdr2 = [np.sum(np.power(d, 2)) for d in dr]

    dmdr2dr = mdr2[1]*dr[0] - mdr2[0]*dr[1]

    drxdr = np.cross(dr[1], dr[0])

    num = np.cross(dmdr2dr, drxdr)
    den = 2*np.sum(np.power(drxdr, 2))

    return ri + (num/den)

print("Constructing the Finite-Volume Operator...")
milestones = np.arange(10) * math.ceil(Ntri/10)
for e in range(Ntri):
    if np.sum(e == milestones) != 0 and e > 0:
        print("{:d}%".format(math.floor(100*(e/Ntri))))

    tri = np.reshape(T[e, :], (3, 1))
    tri_nodes = np.reshape(P[tri, :], (3, 2))

    # Calculate location of the circumcenter of this triangle
    rc = find_circumcenter(tri_nodes)[:2]

    # Loop through each node in the triangle, adding the contribution to L and A
    for qi in range(3):
        i = tri[qi]
        nns = np.array([qj for qj in range(3) if qj != qi])

        rci = rc - tri_nodes[qi, :]

        # Loop through each nearest neighbor node
        for qj in nns:
            j = tri[qj]

            # Compute wij, lij
            rji = tri_nodes[qj, :] - tri_nodes[qi, :]

            lij = math.sqrt(np.sum(np.power(rji, 2)))
            wije = math.sqrt(np.sum(np.power((rci-0.5*rji), 2)))
            phiij = wije/lij

            # Update L
            L[i, j] = L[i, j].toarray()[0] + phiij
            L[i, i] = L[i, i].toarray()[0] - phiij

            # Add contribution to Aie
            A[i] = A[i] + 0.5*(lij*wije)


print("Finite-volume operator constructed.")

print("Finding boundary nodes and applying boundary conditions...")

# Removes a row from an LIL sparse matrix
def rem_row(mat, i):
    if not isinstance(mat, sparse.lil_matrix):
        raise ValueError("works only for LIL format -- use .tolil() first")
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - 1, mat._shape[1])

# Removes a column rom an LIL sparse matrix
def rem_col(mat, j):
    if not isinstance(mat, sparse.lil_matrix):
        raise ValueError("works only for LIL format -- use .tolil() first")
    if j < 0:
        j += mat.shape[1]

    if j < 0 or j >= mat.shape[1]:
        raise IndexError('column index out of bounds')

    rows = mat.rows
    data = mat.data
    for i in range(mat.shape[0]):
        pos = bisect_left(rows[i], j)
        if pos == len(rows[i]):
            continue
        elif rows[i][pos] == j:
            rows[i].pop(pos)
            data[i].pop(pos)
            if pos == len(rows[i]):
                continue
        for pos2 in range(pos, len(rows[i])):
            rows[i][pos2] -= 1

    mat._shape = (mat._shape[0], mat._shape[1] - 1)

# A parameteric function providing the y-values of the bottom of the channel for a given x
def bot_bdry(x):
    if (x > 5 or x < -5):
        if (x < 0): x = x + 5
        if (x > 0): x = x - 5
        return -math.sqrt(100-(x**2))
    else:
        return -10

# Calculates the euclidean distance between two vectors
def dist(v1, v2):
    dv = v1-v2
    return math.sqrt(dv[0]**2 + dv[1]**2)

# Returns 1 if a node is on the boundary of the channel, and 0 otherwise.
def on_boundary(r, top, eps=1e-10):
    return dist(r, [r[0], bot_bdry(r[0])]) <= eps  or (r[1] >= -eps and top)

# Apply no-slip condition to the boundaries
offset = 0
bdry_nodes = list()
for i in range(Npts):
    if on_boundary(P[i, :], use_top, eps=1e-7):
        bdry_nodes.append(i)
        if 1:
            rem_row(L, i-offset)
            rem_col(L, i-offset)
            A = np.delete(A, i-offset)
            offset = offset+1

# Solve discrete equation
print("Solving the discretized equation...")
g = -R*pz*A

L = sparse.csc_matrix(L) # Convert to CSC so we can solve the equation
v = linalg.spsolve(L, g) # Solve the equation

# Insert 0 for the velocity at the boundary nodes
for i in bdry_nodes:
    if 1:
        if i < len(v):
            v = np.insert(v, i, 0)
        else:
            v = np.append(v, 0)

print("Plotting the results...")

triang = triangle_mod.Triangulation(P[:, 0], P[:, 1], triangles=T)

fig1 = plt.figure(dpi=400)
ax1 = fig1.subplots()
ax1.set_aspect('equal')
tpc = ax1.tripcolor(triang, v, shading='flat')
fig1.colorbar(tpc)
ax1.set_title('Velocity [um/s] dist. in microfluidic channel - %s' % sim_name)
ax1.set_xlabel("x [um]")
ax1.set_ylabel("y [um]")
plt.show()
