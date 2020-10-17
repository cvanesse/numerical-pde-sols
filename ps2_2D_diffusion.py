# The general 2D FDTD diffusion simulator code for ECE 570 problem set #2

## IMPORT MODULES
from finite_difference_methods import *
from matplotlib import pyplot as plt
import time

# For timing the simulation
start_time = time.time()

# Spatial domain
dx = 1e-6
dy = 1e-6
X = 10e-6
Y = 10e-6

Nx = math.floor(X / dx)
dx = X / Nx
Ny = math.floor(Y / dy)
dy = Y / Ny

# Time settings
dt = 1e-19 # Timestep in seconds [Will be updated according to CFL condition if too low]
Nt = 500000 # Number of timesteps [Takes priority over T if provided]

# Initial conditions [Just a point in the center]
x_init = 5e-6
y_init = 5e-6

# Diffusion parameter
D = 50

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

dt_max = ((4/3) / D) * (np.sum(np.power(domain['h'], -2)) ** (-1)) # N-Dimensional CFL Condition
if dt > dt_max or dt < 0:
    print("WARNING: Input timestep {:0.2e} is larger than allowed by stability condition (or negative). Updating dt...".format(dt))
    dt = dt_max

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

x_init = math.floor(x_init/domain['h'][1])
y_init = math.floor(y_init/domain['h'][0])
u[0][y_init, x_init] = 1
u[1][y_init, x_init] = 1

# Construct stepping operator [M]
laplacian = cd_1d_matrix_ND_v2(2, 0, domain) + cd_1d_matrix_ND_v2(2, 1, domain)
M = 2 * dt * D * laplacian

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
    u = [i.reshape([domain['size']], order="F") for i in u]
    u[2] = M.dot(u[1]) + u[0] # Apply stepping operator
    u = [i.reshape(domain['shape'], order="F") for i in u]

    # Update solution for the next timestep
    u[0] = np.copy(u[1])
    u[1] = np.copy(u[2])

print("------------------------")
print("-- Plotting Results...")

Y = np.arange(domain['shape'][0])*domain['h'][0]
X = np.arange(domain['shape'][1])*domain['h'][1]

X, Y = np.meshgrid(X, Y)

print("Plotting 2D Colormap...")
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
