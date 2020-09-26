# Calculates the resonant frequencies of a beam (ps#1, p1.b)
# Collin VanEssen, Sept 22nd, 2020

### Import packages
from finite_difference_methods import *
from matplotlib import pyplot as plt

### Config

L = 5e-6 # Length of the beam [m]
w = t = 200e-9 # Width & Thickness for the rectangular beam [m]
rho0 = 2300 # Mass density for silicon [kg/m^3]
E = 1.85e11 # Young's modulus for silicon [Pa]

dx = 0.1e-6 # The size of the 1D mesh
k = 3 # The number of modes to calculate

### Script

N = math.floor(L / dx)
A = w*t
D = E*(w*(math.pow(t, 3)))/12

# Construct a 4-th derivative operator with 2 fictitious nodes on each side
M = cd_1d_matrix(4, N + 4, dx)

# Apply 0th & 1st order derivative homogenous BCs on the left boundary
M = apply_1d_homogenous_bcs(M, [1, 0], 0)

# Apply 2nd & 3rd order derivative homogenous BCs on the right boundary
M = apply_1d_homogenous_bcs(M, [2, 3], 1)

# Find the low-frequency eigenmodes
[w2d_A, y] = linalg.eigs(M/rho0, k=k, which="SM") # which="SM" ensures that the low-frequency modes are selected.
w2d_A = np.real(w2d_A)
y = np.real(y)

# Turn eigenvalues into frequencies
f = (0.5/math.pi) * np.sqrt(w2d_A / A * D )

# Plot the results
x = np.linspace(0, L, num=N)

for mid in range(k):
    plt.plot(x, np.real(y[:, mid]), label=("%f MHz" % (f[mid]/1e6)))

plt.title("First three eigenmodes of cantilever beam")
plt.xlabel("x [m]")
plt.ylabel("Y(x) [m]")
plt.legend()
plt.show()
