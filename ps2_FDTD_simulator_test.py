# The general 2D FDTD simulator code for ECE 570 problem set #2

## IMPORT MODULES
from finite_difference_methods import *
from matplotlib import pyplot as plt

## SIMULATION CONFIG

# Physical constants
c = 2.997925e8

# Physical variables
n = 1 # Refractive index of the domain

# Spatial domain
dx = 0.05e-6
dy = 0.05e-6
X = 15e-6
Y = 20e-6

Nx = math.floor(X / dx)
dx = X / Nx
Ny = math.floor(Y / dy)
dy = Y / Ny

# Time settings
dt = -0.5e-16 # Timestep in seconds [Will be updated according to CFL condition if too low]
Nt = 100 # Number of timesteps [Takes priority over T if provided]

# Source settings (Specific to simulation #1)
wl = 1e-6 # Pulse wavelength (in m)
omega = 2*math.pi*c/wl # Pulse frequency (in rads/s)
w = 8e-15 # Pulse width (in seconds)
T0 = 4e-15 # Pulse time (in seconds)
source_position = np.array([5.6e-6, 5e-6]) # The source position in m
A = 1 # Pulse amplitude

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

dt_CFL = (n/c) * (np.sum(np.power(domain['h'], -2)) ** (-0.5)) # N-Dimensional CFL Condition
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

# Construct stepping operator [M]
laplacian = cd_1d_matrix_ND_v2(2, 0, domain) + cd_1d_matrix_ND_v2(2, 1, domain)
#laplacian = cd_1d_matrix_ND(2, 0, domain) + cd_1d_matrix_ND(2, 1, domain)
M = 2*sparse.eye(domain['size']) + (c**2/n**2)*(dt**2) * laplacian

# Calculate source information (Specific to simulation 1)
#source_node = math.floor(get_node_number(source_position, domain))
source_node = np.round(source_position / domain['h']).astype('int64')

print("Domain: " + str(domain))
print("Operator Size: {:d}, {:d}".format(np.shape(M)[0], np.shape(M)[1]))
print("Solution Size: {:d}, {:d}".format(np.shape(u[2])[0], np.shape(u[2])[1]))

np.set_printoptions(precision=1, suppress=True)

print("------------------------")
print("-- Running FDTD simulation...")

for i in range(Nt):
    t = dt*i # The time (in seconds)

    # First, apply the stepping operator to the internal nodes
    u = [i.reshape([domain['size']], order="F") for i in u]
    u[2] = M.dot(u[1]) - u[0] # Apply stepping operator
    u = [i.reshape(domain['shape'], order="F") for i in u]

    # Apply the radiating boundary conditions for each boundary
    u[2] = apply_radiating_BC(u[2], u[1], 0, 0, n / (c * dt), domain) # Bottom boundary
    u[2] = apply_radiating_BC(u[2], u[1], 1, 0, n / (c * dt), domain) # Left boundary
    u[2] = apply_radiating_BC(u[2], u[1], 1, 1, n / (c * dt), domain) # Right boundary
    u[2] = apply_radiating_BC(u[2], u[1], 0, 1, n / (c * dt), domain)  # Top boundary

    # Set the source nodes to the appropriate value
    u[2][source_node[0], source_node[1]] = A*math.exp(-(((t-T0)/(w/2))**2))*math.sin(omega*t)

    # Update solution for the next timestep
    u[0] = np.copy(u[1])
    u[1] = np.copy(u[2])

print("------------------------")
print("-- Plotting Results...")

Y = np.arange(domain['shape'][0])*domain['h'][0]
X = np.arange(domain['shape'][1])*domain['h'][1]

X, Y = np.meshgrid(X, Y)
fig = plt.figure()
ax = plt.axes()
ax.contourf(X, Y, u[2], 100)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

exit()

fig = plt.figure()
plt.plot(X, u[2][:, 0])
plt.show()


X, Y = np.meshgrid(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contourf(X, Y, u[2], 100)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
plt.ion()
ax.view_init(50, -45)
plt.show()
