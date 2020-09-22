# Calculates the resonant frequencies of a beam (ps#1, p1.b)

### Import packages
from scipy import sparse
from scipy.sparse import linalg
import numpy as np
import math
from matplotlib import pyplot as plt

### Config

L = 5e-6 # Length of the beam [m]
w = t = 200e-9 # Width & Thickness for the rectangular beam [m]
rho = 2300 # Mass density for silicon [kg/m^3]
E = 1.85e11 # Young's modulus for silicon [kPa]

dx = 0.1e-6 # The size of the 1D mesh
k = 3 # The number of modes to calculate

### Function Definitions

# Builds the matrix based on the problem definition for 1.b
def build_A(dx, N):
    # Construct fourth-order derivative matrix
    A = sparse.diags([1, -4, 6, -4, 1], [-2, -1, 0, 1, 2], shape=(N, N))
    A = sparse.csr_matrix(A)

    # Fill in x=0 bdry conditions
    A[0, 0] = A[0, 0] + 1

    # Fill in x=L bdry conditions
    A[N-3:N, N-3:N] = A[N-3:N, N-3:N] + sparse.csr_matrix([[0, 0, 0], [0, -1, 2], [1, 0, -4]])

    return A / dx

### Script

N = math.floor(L / dx)
A = w*t
D = E*(w*(math.pow(t, 3)))/12

# Build the matrix and solve the eigenvalue problem
M = build_A(dx, N)
[w2d_rhoA, y] = linalg.eigs(M, k=k, which="SR") # which="SR" ensures that the low-frequency modes are selected.
w2d_rhoA = np.real(w2d_rhoA)
y = np.real(y)

# Turn eigenvalues into frequencies
f = (0.5/math.pi) * np.sqrt(w2d_rhoA * rho * A / D )

# Plot the results
x = np.linspace(0, L, num=N)
for mid in range(k):
    plt.plot(x, np.real(y[:, mid]), label=("%f KHz" % (f[mid]/1000)))

plt.title("First three eigenmodes of cantilever beam")
plt.xlabel("x [m]")
plt.ylabel("Y(x) [m]")
plt.legend()
plt.show()
