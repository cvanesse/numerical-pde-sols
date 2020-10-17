# The general 2D FDTD simulator code for ECE 570 problem set #2

## IMPORT MODULES
from finite_difference_methods import *
from matplotlib import pyplot as plt
import time

# For timing the simulation
start_time = time.time()

## SIMULATION CONFIG

# Physical constants
c = 2.997925e8

# Physical variables
n_bg = 1 # Refractive index of the domain
n_barrier = 10 # Refractive index of the barrier
x_barrier = [9.5e-6, 10e-6]
slot_width = 5e-6

# Spatial domain
dx = 0.05e-6
dy = 0.05e-6
X = 25e-6
Y = 30e-6

Nx = math.floor(X / dx)
dx = X / Nx
Ny = math.floor(Y / dy)
dy = Y / Ny

# Time settings
dt = -1e-16 # Timestep in seconds [Will be updated according to CFL condition if too low]
Nt = 1000 # Number of timesteps [Takes priority over T if provided]

# Source settings (Specific to simulation #1)
wl = 1e-6 # Pulse wavelength (in m)
omega = 2*math.pi*c/wl # Pulse frequency (in rads/s)
A = 1 # Pulse amplitude
source_x = [1e-6]
source_y = [0.5e-6, 29e-6]

# Screen settings
x_screen = 20e-6

## CODE
print("------------------------")
print("-- Parsing Inputs...")
inputs = locals()

# Index the domain for use throughout the code.
domain = {
    "shape": np.array([Ny, Nx]),
    "size": np.prod(np.array([Ny, Nx])),
    "h": np.array([dy, dx])
}

print("Ymax: \t\t\t {:0.2e}".format(Y))
print("dy: \t\t\t {:0.2e}".format(dy))
print("Ny: \t\t\t {:d}".format(Ny))
print("Xmax: \t\t\t {:0.2e}".format(X))
print("dx: \t\t\t {:0.2e}".format(dx))
print("Nx: \t\t\t {:d}".format(Nx))

dt_CFL = (min([n_bg, n_barrier])/c) * (np.sum(np.power(domain['h'], -2)) ** (-0.5)) # N-Dimensional CFL Condition
if dt > dt_CFL or dt < 0:
    print("WARNING: Input timestep {:0.2e} is larger than allowed by CFL condition (or negative). Updating dt...".format(dt))
    dt = dt_CFL

print("Timestep: \t\t\t {:0.2e} seconds".format(dt))

if 'Nt' not in inputs:
    if 'T' in inputs:
        Nt = math.ceil(T / dt) # The number of timesteps to simulate
    else:
        print("Error: No simulation time provided!")
        exit()
else:
    T = dt*Nt

print("Simulation Time: \t {:0.2e} seconds".format(T))

print("------------------------")
print("-- Initializing Solutions and Operators...")

# Allocate space for solution vectors
u = [np.zeros(domain["shape"]),
     np.zeros(domain["shape"]),
     np.zeros(domain["shape"])] # Solution is a list of arrays

# Construct refractive index matrix
n = n_bg*np.ones(domain["shape"]) # Start in cartesian coordinates

idx_barrier = np.round(x_barrier/domain['h'][1]).astype('int64')
y_barrier = 15e-6 + np.array([-slot_width/2, slot_width/2])
idy_barrier = np.round(y_barrier/domain['h'][0]).astype('int64')
n[:np.min(idy_barrier), np.min(idx_barrier):np.max(idx_barrier)] = n_barrier
n[np.max(idy_barrier):, np.min(idx_barrier):np.max(idx_barrier)] = n_barrier

n = np.reshape(n, [domain["size"]], order="F")
n = np.power(n, -2)
n = sparse.diags(n, format="csr")

# Construct stepping operator [M]
laplacian = cd_1d_matrix_ND_v2(2, 0, domain) + cd_1d_matrix_ND_v2(2, 1, domain)
M = 2*sparse.eye(domain['size']) + c**2*dt**2 * n.dot(laplacian)

# Calculate source information (Specific to simulation 1)
source_idx = np.round(source_x / domain['h']).astype('int64')
source_idy = np.round(source_y / domain['h']).astype('int64')
source_idy = [np.min(source_idy), np.max(source_idy)]

print("Domain: " + str(domain))
print("Operator Size: {:d}, {:d}".format(np.shape(M)[0], np.shape(M)[1]))
print("Solution Size: {:d}, {:d}".format(np.shape(u[2])[0], np.shape(u[2])[1]))

np.set_printoptions(precision=1, suppress=True)

print("------------------------")
print("-- Running FDTD simulation...")

milestones = np.arange(10) * math.ceil(Nt/10)
for i in range(Nt):
    if np.sum(i == milestones) != 0 and i > 0:
        print("{:d}%".format(math.ceil(100*(i/Nt))))
    t = dt*i # The time (in seconds)

    # First, apply the stepping operator to the internal nodes
    u = [j.reshape([domain['size']], order="F") for j in u]
    u[2] = M.dot(u[1]) - u[0] # Apply stepping operator
    u = [j.reshape(domain['shape'], order="F") for j in u]

    # Apply the radiating boundary conditions for each boundary
    ## (I'm just using n_bg for my boundary conditions as an approximation
    u[2] = apply_radiating_BC(u[2], u[1], 0, 0, n_bg / (c * dt), domain) # Bottom boundary
    u[2] = apply_radiating_BC(u[2], u[1], 1, 0, n_bg / (c * dt), domain) # Left boundary
    u[2] = apply_radiating_BC(u[2], u[1], 1, 1, n_bg / (c * dt), domain) # Right boundary
    u[2] = apply_radiating_BC(u[2], u[1], 0, 1, n_bg / (c * dt), domain)  # Top boundary

    # Set the source nodes to the appropriate value
    u[2][np.min(source_idy):np.max(source_idy), source_idx] = A*math.sin(omega*t)

    # Update solution for the next timestep
    u[0] = np.copy(u[1])
    u[1] = np.copy(u[2])

print("------------------------")
print("-- Plotting Results...")

Y = np.arange(domain['shape'][0])*domain['h'][0]
X = np.arange(domain['shape'][1])*domain['h'][1]

print("Plotting cross-section at the screen vs. theoretical results...")
beta = (math.pi*slot_width/wl)*np.sin(np.arctan((Y - 15e-6)/(x_screen - np.max(x_barrier))))
I_fraunhofer = np.power(np.sin(beta)/beta, 2.0)
x_screen = math.floor(x_screen/domain['h'][1])
I = np.power(u[2][:, x_screen], 2)
I_fraunhofer = I_fraunhofer * np.max(I)
fig = plt.figure()
plt.plot(Y, I, label="Numerical Results")
plt.plot(Y, I_fraunhofer, label="Theoretical Results")
plt.legend()
plt.show()

print("Plotting 2D Colormap...")
X, Y = np.meshgrid(X, Y)
fig = plt.figure()
ax = plt.axes()
ax.contourf(X, Y, u[2], 100)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

print("Plotting 3D Surface Rendering...")
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contourf(X, Y, u[2], 100)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
plt.ion()
ax.view_init(50, -45)
plt.show()

print("Done! Runtime: {:.2f} seconds".format(time.time() - start_time))
