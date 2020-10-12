# The general 2D FDTD simulator code for ECE 570 problem set #2

## IMPORT MODULES
from finite_difference_methods import *
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


## SIMULATION CONFIG

# Physical constants
c = 2.997925e8

# Physical variables
n = 1 # Refractive index of the domain

# Spatial domain
dx = 0.05e-6
dy = 0.05e-6
X = 10e-6
Y = 10e-6

Nx = math.floor(X / dx)
dx = X / Nx
Ny = math.floor(Y / dy)
dy = Y / Ny

# Time settings
dt = -1 # Timestep in seconds [set to -1 to use max allowable by CFL condition]
T = -1 # Simulation time in seconds
Nt = 175 # Number of timesteps [Takes priority over T if provided]

# Source settings (Specific to simulation #1)
wl = 1e-6 # Pulse wavelength (in m)
omega = 2*math.pi*c/wl # Pulse frequency (in rads/s)
w = 8e-15 # Pulse width (in seconds)
T0 = 4e-15 # Pulse time (in seconds)
source_position = np.array([2.5e-6, 5e-6]) # The source position in m

## CODE

inputs = locals()

# Index the domain for use throughout the code.
domain = {
    "shape": np.array([Nx, Ny]),
    "size": np.prod(np.array([Nx, Ny])),
    "h": np.array([dx, dy])
}

if dt == -1:
    # Calculate timestep based on CFL condition
    dt = (n/c) * (np.sum(np.power(domain['h'], -2)) ** (-0.5)) # Generalized N-Dimensional CFL Condition

if 'Nt' not in inputs:
    if 'T' in inputs:
        Nt = math.ceil(T / dt) # The number of timesteps to simulate
    else:
        print("Error: No simulation time provided!")
        exit()
else:
    T = dt*Nt

# Allocate space for solution vectors
u = [np.zeros(domain["shape"]),
     np.zeros(domain["shape"]),
     np.zeros(domain["shape"])] # Solution is a list of arrays

# Construct stepping operator [M]
laplacian = cd_1d_matrix_ND(2, 0, domain) + cd_1d_matrix_ND(2, 1, domain)
M = 2*cd_1d_matrix_ND(0, 0, domain) + (c**2/n**2)*(dt**2) * laplacian

# Calculate source information (Specific to simulation 1)
source_node = math.floor(get_node_number(source_position, domain))

for i in range(Nt):
    t = dt*i # The time (in seconds)

    # First, apply the stepping operator to the internal nodes
    u[2] = M.dot(u[1][:].flat) - u[0][:].flat # Apply stepping operator
    u[2] = np.reshape(u[2][:], domain['shape']) # Reshape into matrices for simpler indexing.

    # Apply the radiating boundary conditions for each boundary
    apply_radiating_BC(u[2], u[1], 0, 0, n / (c * dt), domain) # Left boundary
    apply_radiating_BC(u[2], u[1], 0, 1, n / (c * dt), domain) # Right boundary
    apply_radiating_BC(u[2], u[1], 1, 0, n / (c * dt), domain) # Top boundary
    apply_radiating_BC(u[2], u[1], 1, 1, n / (c * dt), domain) # Bottom boundary

    # Set the source nodes to the appropriate value
    u[2].flat[source_node] = math.exp(-(((t-T0)/(w/2))**2))*math.sin(omega*t)

    # Update solution for the next timestep
    u[0] = np.copy(u[1])
    u[1] = np.copy(u[2])

X = np.arange(domain['shape'][0])*domain['h'][0]
Y = np.arange(domain['shape'][1])*domain['h'][1]

fig = plt.figure()
plt.plot(X, u[2][:, 0])
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, u[2], 50)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
plt.ion()
plt.show()


