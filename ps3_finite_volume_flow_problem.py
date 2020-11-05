# Finite-Volume solver for the flow problem
from scipy.io import loadmat
import numpy as np
from scipy import sparse
import math
from scipy.sparse import linalg
from matplotlib import pyplot as plt
from matplotlib import tri as triangle_mod

## Load simulation grid from provided .mat file
mesh_data = loadmat("data/channel_mesh.mat")
P = np.array(mesh_data['P']) # The node coordinates in the computational domain
T = np.array(mesh_data['T'])-1 # The node indexes of each triangle (converted to python indexing)

# Determine the number of nodes in the simulation domain
Npts = np.shape(P)[0]
Ntri = np.shape(T)[0]

## Simulation Parameters
pz = 1 # Pressure gradient in z (transport) direction
R  = 1 # Reynold's number

# Initialize variables for the discrete equation
L = sparse.lil_matrix((Npts, Npts))
A = np.zeros((Npts, 1))

def bot_bdry(x):
    if (x > 5 or x < -5):
        if (x < 0): x = x + 5
        if (x > 0): x = x - 5
        return -math.sqrt(100-x**2)
    else:
        return -10

def on_boundary(r, eps=1e-10):
    return (r[1]==0 or math.abs(r[1]-bot_bdry(r[0])) < eps)

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

# Loop through each triangle in the domain
ide = [0, 1, 2]

print("Constructing the Finite-Volume Operator...")
milestones = np.arange(10) * math.ceil(Ntri/10)
for e in range(Ntri):
    if np.sum(e == milestones) != 0 and e > 0:
        print("{:d}%".format(math.floor(100*(e/Ntri))))

    tri = np.reshape(T[e, :], (3, 1))
    tri_nodes = np.reshape(P[tri, :], (3, 2))

    # Calculate location of the circumcenter of this triangle
    rc = find_circumcenter(tri_nodes)[:2]

    # Loop through each node in the triangle, adding the contribution to L and g
    for qi in ide:
        i = tri[qi]
        Lii = L[i,i].toarray()[0]
        nns = np.array([qj for qj in ide if qj != qi])


        for qj in nns:
            j = tri[qj]

            # Compute wij, lij
            rji = tri_nodes[qj, :] - tri_nodes[qi, :]
            rci = rc - tri_nodes[qi, :]

            lij = np.sum(np.power(rji, 2))
            wij = np.sum(np.power((rci-0.5*rji), 2))
            phiij = wij/lij

            # Update L
            L[i, j] = phiij
            Lii = Lii - phiij

            # Add contribution to Aie
            A[i] = A[i] + 0.5*(lij*wij)

        L[i, i] = Lii

print("Finite-volume operator constructed.")

print("Finding boundary nodes and applying boundary conditions...")

def bot_bdry(x):
    if (x > 5 or x < -5):
        if (x < 0): x = x + 5
        if (x > 0): x = x - 5
        return -math.sqrt(100-x**2)
    else:
        return -10

def on_boundary(r, eps=1e-10):
    return r[1] >= -eps or abs(r[1]-bot_bdry(r[0])) <= eps

for i in range(Npts):
    if on_boundary(P[i, :], eps=0):
        L[:, i] = 0

# Solve discrete equation
print("Solving the discretized equation...")
g = -R*pz*A

L = sparse.csc_matrix(L)
v = linalg.spsolve(L, g)


print("Plotting the results...")
triang = triangle_mod.Triangulation(P[:, 0], P[:, 1], triangles=T)

fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
tpc = ax1.tripcolor(triang, v, shading='flat')
fig1.colorbar(tpc)
ax1.set_title('tripcolor of Delaunay triangulation, flat shading')
plt.show()

# Display results


