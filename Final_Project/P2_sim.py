# Runs the poisson equation calculation for Problem 2

from os import path
from finite_element_poisson import *
from matplotlib import pyplot as plt
from matplotlib import tri as triangle_mod
from matplotlib import cm, colors

## Physical Information
eps_0 = 8.854187e-12 * 1e6 # Freespace permittivity in um
eps_r_water = 80
eps_r_cell = 50
V_elec1 = 10
V_elec2 = -10

## Script config
#mesh_name = "P2_mesh_extremely_fine_uniform"
#mesh_name = "P2_mesh_very_fine_uniform"
#mesh_name = "P2_mesh_fine_uniform"
mesh_name = "P2_mesh_uniform"
show_mesh_plots = False
show_sparsity = False

curdir = path.dirname(__file__)

# Load the mesh and triangulation data
meshdata = np.load(path.join(curdir, "meshes/" + mesh_name + ".npz"))

# Extract all of the mesh information from the file
P = meshdata["P"]
N = np.shape(P)[0]
T = meshdata["T"]
N_e = np.shape(T)[0]
P_w = meshdata["P_sep"] # Points in water
P_e1 = meshdata["P_e1"] # Points in the first electrode
P_e2 = meshdata["P_e2"] # Points in the second electrode
P_c =  meshdata["P_c"]  # Points in the cell

# Separate the point indices into regions
x_min = np.min(P[:, 0])
x_max = np.max(P[:, 0])
y_min = np.min(P[:, 1])
y_max = np.max(P[:, 1])
eps_select = 1e-5

n_bdry = []
n_corn = []
n_water = []
n_e1 = []
n_e2 = []
n_cell = []
for i in range(N):
    ri = P[i, :]
    if ri[0] in P_e1[:, 0] and ri[1] in P_e1[:, 1]:
        n_e1.append(i)
    elif ri[0] in P_e2[:, 0] and ri[1] in P_e2[:, 1]:
        n_e2.append(i)
    elif ri[0] in P_c[:, 0] and ri[1] in P_c[:, 1]:
        n_cell.append(i)
    else:
        lb = int(abs(ri[0] - x_min) < eps_select)
        rb = int(abs(ri[0] - x_max) < eps_select)
        #bb = int(abs(ri[1] - y_min) < eps_select) # We leave the bottom boundary as neumann
        tb = int(abs(ri[1] - y_max) < eps_select)

        num_boundaries = sum([lb, rb, tb])

        if num_boundaries > 1:
            n_corn.append(i)
        elif num_boundaries:
            n_bdry.append(i)
        else:
            n_water.append(i)

n_bdry = np.array(n_bdry)
n_corn = np.array(n_corn)
n_water = np.array(n_water)
n_e1 = np.array(n_e1)
n_e2 = np.array(n_e2)
n_cell = np.array(n_cell)

if show_mesh_plots:
    plt.scatter(P[n_bdry, 0], P[n_bdry, 1], s=5, color="red", marker=".", label="Boundary")
    plt.scatter(P[n_e1, 0], P[n_e1, 1], s=5, color="black", marker=".", label="Electrode 1")
    plt.scatter(P[n_e2, 0], P[n_e2, 1], s=5, color="grey", marker=".", label="Electrode 2")
    plt.scatter(P[n_cell, 0], P[n_cell, 1], s=5, color="green", marker=".", label="Cell")
    plt.scatter(P[n_water, 0], P[n_water, 1], s=5, color="blue", marker=".", label="Water")
    plt.title("Identified Regions")
    plt.legend()
    plt.show()

# Reorder the indices for clearer ordering
new_point_order = np.hstack((n_corn, n_bdry, n_e1, n_e2, n_cell, n_water))
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
n_e1 = invmap[n_e1]
n_e2 = invmap[n_e2]
n_cell = invmap[n_cell]
n_water = invmap[n_water]
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

# Set the relative permittivity for each point and triangle according to it's location
eps_p = np.ones((N))
eps_e = np.ones((N_e))

eps_p[n_water] *= eps_r_water
eps_p[n_bdry] *= eps_r_water
eps_p[n_corn] *= eps_r_water
eps_p[n_cell] *= eps_r_cell # We don't need to worry about electrodes since those end up being overridden with dirichlet

# Set triangle refractive indices according to *priority*
#   (We assume that any triangle containing a single water point is entirely within the water)
water_nodes = np.hstack((n_corn, n_bdry, n_water, n_e1, n_e2))
for e in range(N_e):
    set=False
    for p in T[e, :]:
        if p in water_nodes:
            eps_e[e] *= eps_r_water
            set=True
            break
    if not set:
        eps_e[e] *= eps_r_cell

# Plot the refractive index for each circumcenter
if show_mesh_plots:
    plt.scatter(C[:, 0], C[:, 1], s=5, c=eps_e, marker=".", label="Circumcenters")
    plt.title("Refractive indices at circumcenters")
    plt.show()

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
V_elecs = np.zeros(np.size(n_e1) + np.size(n_e2))
V_elecs[:np.size(n_e1)]  = V_elec1
V_elecs[-np.size(n_e2):] = V_elec2
n_rods = np.hstack((n_e1, n_e2))
K, b = apply_dirichlet_conditions(K, b, n_rods, water_nodes, V_elecs)

if show_sparsity:
    plt.spy(K, markersize=0.5)
    plt.title("Sparsity of Stiffness Matrix with Dirichlet Conditions")
    plt.show()

# Apply the radiating boundary conditions at boundary nodes
print("Applying Radiating Boundary Conditions...")
K, b = apply_RBCs(K, b, P, T, C, n_bdry, n_corn, eps_p)

print("Solving the Matrix Equation...")
K = K.tocsr()
V = linalg.spsolve(K, b)

print("Plotting the potential distribution...")
triang_out = triangle_mod.Triangulation(P[:, 0], P[:, 1], triangles=T)

fig1, ax1 = plt.subplots(dpi=400)
ax1.set_aspect('equal')
tpc = ax1.tripcolor(triang_out, V, shading='gouraud')
fig1.colorbar(tpc)
ax1.set_title('Potential distribution')
plt.show()

# Calculate and plot the electric field
E = -FE_gradient(V, P, T)

# Calculate the force on the cell
theta = [] # For plotting E vs. theta
E_ccl = []
F_ccl = []
P_ccl = []
F = np.zeros((2))
# Calculate the capacitance by integrating the normal electric field around the small rod
T_maxwell = np.zeros((2, 2))
for e in range(N_e):
    pts = T[e, :]

    # Check which points (if any) are on the cell
    bpts = []  # For edge and corner points
    opts = []  # For internal points
    for p in pts:
        if p in n_cell:
            bpts.append(p)  # bpts will be in CCW order.
        else:
            opts.append(p)

    # Go to the next triangle if this one isn't on the cell boundary
    on_bdry = len(bpts) > 1 and len(opts) == 1
    if not on_bdry:
        continue

    # Calculate the normal vector
    rc = C[e]
    ri = P[bpts[0]]
    rj = P[bpts[1]]
    n = (rc-ri) - 0.5*(ri-rj)
    un = n / np.linalg.norm(n) # Unit normal vector pointing into the triangle

    theta_x = np.arcsin(un[0])
    theta_y = np.arccos(un[1])

    if (theta_y < np.pi/2):
        # Top half of the circle
        if (theta_x > 0):
            # Right top
            theta.append(0.5*(theta_x+theta_y))
        else:
            # Left top
            theta.append(0.5*((2*np.pi + theta_x) + (2*np.pi - theta_y)))
    else:
        # Bottom half of the circle
        if (theta_x > 0):
            # Right bot
            theta.append(0.5 * ((np.pi - theta_x) + theta_y))
        else:
            # Left bot
            theta.append(0.5 * ((np.pi - theta_x) + (2*np.pi - theta_y)))

    E_ccl.append(E[e, :])
    P_ccl.append(C[e, :])

    T_maxwell[0, 0] = 0.5 * (E[e, 0] ** 2 - E[e, 0] ** 2)
    T_maxwell[0, 1] = E[e, 0]*E[e, 1]
    T_maxwell[1, 0] = E[e, 0]*E[e, 1]
    T_maxwell[1, 1] = -T[0, 0]
    T_maxwell = T_maxwell * eps_r_water*eps_0

    F_ccl.append(T_maxwell.dot(un))

    l = np.linalg.norm(ri - rj)  # Length of the boundary
    F = F+l*F_ccl[-1]

print("----------------------------------------------")

# Finally, multiply by epsilon to get charge
print("Force on the cell: (%0.2e, %0.2e) Newtons" % (F[0], F[1]))

theta = np.array(theta)
sortmap = np.argsort(theta)
E_ccl = np.array(E_ccl)
F_ccl = np.array(F_ccl)
P_ccl = np.array(P_ccl)

theta = theta[sortmap]
E_ccl = E_ccl[sortmap, :]
F_ccl = F_ccl[sortmap, :]
P_ccl = P_ccl[sortmap, :]

# Plot the global position components vs. theta
plt.subplots(dpi=400)
plt.plot(theta, P_ccl[:, 0], label="x")
plt.plot(theta, P_ccl[:, 1], label="y")
plt.title("Global coordinates components versus angle")
plt.legend()
plt.show()

# Plot the electric field components vs. theta
plt.subplots(dpi=400)
plt.plot(theta, E_ccl[:, 0], label="Ex")
plt.plot(theta, E_ccl[:, 1], label="Ey")
plt.title("Electric field [V/um] components versus angle")
plt.legend()
plt.show()

# Plot the force components vs. theta
plt.subplots(dpi=400)
plt.plot(theta, F_ccl[:, 0], label="Fx")
plt.title("Force density in x [N/um] components versus angle")
plt.legend()
plt.show()
plt.subplots(dpi=400)
plt.plot(theta, F_ccl[:, 1], label="Fy")
plt.title("Force density in y [N/um] components versus angle")
plt.legend()
plt.show()

# Normalize E for scaling
magE = np.linalg.norm(E, axis=1)
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
cb.ax.set_ylabel("log(|E [V/um]|), saturated at 0", rotation=90)
plt.show()