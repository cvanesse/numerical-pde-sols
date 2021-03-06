# Runs the simulation for problem 1
import numpy as np
from os import path
from finite_element_poisson import *
from matplotlib import pyplot as plt
from matplotlib import tri as triangle_mod
from matplotlib import cm, colors
from scipy.spatial import Delaunay

mesh_name = "P1_very_fine_mesh_uniform"
#mesh_name = "P1_fine_mesh_uniform"
#mesh_name = "P1_mesh"
#mesh_name = "P1_fine_mesh_nonuniform"
#mesh_name = "PP_fine"
#mesh_name = "PP_coarse"
#mesh_name = "PP_very_fine"

## Physical information
eps_0 = 8.854187e-12 * 1e3 # Freespace permittivity in mm
eps_r = 1
V_r1 = 1
V_r2 = -1

## Script config
show_mesh_plots = False
show_sparsity = False
E_quiver = True
calc_distance = False
use_RBCs = True

curdir = path.dirname(__file__)

# Load the mesh and triangulation data
meshdata = np.load(path.join(curdir, "meshes/"+ mesh_name +".npz"))

# Load all of the mesh information
P = meshdata["P"]
N = np.shape(P)[0]
T = meshdata["T"]
N_e = np.shape(T)[0]
p_fs = meshdata["P_sep"]
p_r1 = meshdata["P_r1"]
p_r2 = meshdata["P_r2"]

if (calc_distance):
    print("Calculating mean NN distance of the mesh...")
    mean_point_distance = calculate_characteristic_distance(P, num_per_calc=15000, verbose=False)
    print("Mean Point Distance: %0.4e" % mean_point_distance)

x_min = np.min(P[:, 0])
x_max = np.max(P[:, 0])
y_min = np.min(P[:, 1])
y_max = np.max(P[:, 1])
eps_select = 1e-5 # The minimum distance from the boundary for a point to be considered a boundary node.

# Calculate the indices of different sections of the domain
#   (For easier indexing/application of BCs)
n_bdry = []
n_corn = []
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
        lb = int(abs(ri[0] - x_min) < eps_select)
        rb = int(abs(ri[0] - x_max) < eps_select)
        bb = int(abs(ri[1] - y_min) < eps_select)
        tb = int(abs(ri[1] - y_max) < eps_select)

        num_boundaries = sum([lb, rb, bb, tb])

        if num_boundaries > 1:
            n_corn.append(i)
        elif num_boundaries:
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
new_point_order = np.hstack((n_corn, n_bdry, n_r1, n_r2, n_cent))
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
n_corn = invmap[n_corn]
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
max_sum = 0
for i in range(N):
    max_sum = max([max_sum, np.abs(np.sum(K[i, :]))])

eps_test = 1e-25
valid = max_sum <= eps_test
if not valid:
    print("ERROR: Invalid Stiffness Matrix - Sum of rows are not all 0.")
    print("Maximum Row sum is: %0.2e" % max_sum)

if show_sparsity:
    plt.spy(K, markersize=0.5)
    plt.title("Sparsity of Stiffness Matrix Before BC Application")
    plt.show()

# Apply the dirichlet conditions (rod voltages) to the FE equation
print("Applying Dirichlet Conditions...")
V_rods = np.zeros(np.size(n_r1) + np.size(n_r2))
V_rods[:np.size(n_r1)]  = V_r1
V_rods[-np.size(n_r2):] = V_r2
n_rods = np.hstack((n_r1, n_r2))
K, b = apply_dirichlet_conditions(K, b, n_rods, n_cent, V_rods)

if show_sparsity:
    plt.spy(K, markersize=0.5)
    plt.title("Sparsity of Stiffness Matrix with Dirichlet Conditions")
    plt.show()

# Apply the radiating boundary conditions at boundary nodes
if use_RBCs:
    print("Applying Radiating Boundary Conditions...")
    K, b = apply_RBCs(K, b, P, T, C, n_bdry, n_corn, eps_p)
elif use_RBCs is None:
    print("Leaving boundaries as homogenous neumann conditions...")
else:
    print("Applying Dirichlet (V=0) at Domain Boundary")
    V_bdry = np.zeros(np.size(n_bdry))
    K, b = apply_dirichlet_conditions(K, b, n_bdry, n_cent, V_bdry)

print("Solving the Matrix Equation...")
K = K.tocsr()
V = linalg.spsolve(K, b)

print("Plotting the potential distribution...")
triang_out = triangle_mod.Triangulation(P[:, 0], P[:, 1], triangles=T)

fig1, ax1 = plt.subplots(dpi=400)
ax1.set_aspect('equal')
tpc = ax1.tripcolor(triang_out, V, cmap="turbo", shading='gouraud')
fig1.colorbar(tpc)
ax1.set_title('Potential distribution')
plt.show()

# Calculate the electric field
E = -FE_gradient(V, P, T) #/2#/1.5

# Calculate the capacitance by integrating the normal electric field around the small rod
Q = 0
L = 0
for e in range(N_e):
    pts = T[e, :]

    # Check which points (if any) are on the small wire
    bpts = []  # For edge and corner points
    opts = []  # For internal points
    for p in pts:
        if p in n_r2:
            bpts.append(p)  # bpts will be in CCW order.
        else:
            opts.append(p)

    # Go to the next triangle if this one isn't on the boundary
    on_bdry = len(bpts) > 1 and len(opts) == 1
    if not on_bdry:
        continue

    # Calculate the normal vector
    rc = C[e]
    ri = P[bpts[0]]
    rj = P[bpts[1]]
    n = (rc-ri) - 0.5*(ri-rj)
    un = n / np.linalg.norm(n) # Unit normal vector pointing into the triangle

    l = np.linalg.norm(ri - rj)  # Length of the boundary

    En = np.dot(E[e, :], un)

    L += l
    Q += En*l

print("----------------------------------------------")

# Finally, multiply by epsilon to get charge
Q = eps_0*1e-3*Q

print("Charge on small rod: %0.2e Coulombs" % Q)
print("Voltage on small rod: %0.2e Volts" % V_r2)
print("Length of integral: %0.2e mm" % L)
print("Capacitance (Numerical Result): %0.2e Farads" % (Q/(V_r2)))

C_analytic = 2*np.pi*(eps_0*1e-3)/math.log((0.35**2)/(0.1*0.05))
print("Capacitance (Analytic Result): %0.2e Farads" % C_analytic)

# Plot Electric field
magE = np.linalg.norm(E, axis=1)

if not E_quiver:
    T_centers = Delaunay(C)
    triang_E = triangle_mod.Triangulation(C[:, 0], C[:, 1], triangles=T_centers.simplices)

    fig1, ax1 = plt.subplots(dpi=400)
    ax1.set_aspect('equal')
    tpc = ax1.tripcolor(triang_E, magE, cmap="turbo", shading='gouraud')
    fig1.colorbar(tpc)
    if (calc_distance):
        ax1.set_title('Electric Field Magnitude \n Mean Point Distance %0.4e mm' % mean_point_distance)
    else:
        ax1.set_title('Electric Field Magnitude')
    plt.show()
else:
    eps_plot = 1e-10
    nonzeroE = np.where(magE > eps_plot)
    zeroE = np.where(magE <= eps_plot)
    E[nonzeroE, 0] = E[nonzeroE, 0] / magE[nonzeroE]
    E[nonzeroE, 1] = E[nonzeroE, 1] / magE[nonzeroE]

    magE[nonzeroE] = np.log10(magE[nonzeroE])
    magE[np.where(magE<0)] = 0
    magE[zeroE] = 0

    norm = colors.Normalize()
    norm.autoscale(magE)
    cm = cm.get_cmap("turbo")

    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])

    fig1, ax1 = plt.subplots(dpi=400)
    ax1.set_aspect('equal')
    quiv_scale = np.max(E)*75
    plt.quiver(C[:, 0], C[:, 1], E[:, 0], E[:, 1], color=cm(magE), scale=quiv_scale, headwidth=2, headlength=3)
    ax1.set_title('Electric Field Lines')
    cb = fig1.colorbar(sm)
    cb.ax.set_ylabel("log(|E [V/mm]|), saturated at 0", rotation=90)
    plt.show()