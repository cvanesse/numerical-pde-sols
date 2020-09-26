# Calculates the resonant frequencies of a beam (ps#1, p1.b)
# Collin VanEssen, Sept 22nd, 2020

### Import packages
from finite_difference_methods import *
from matplotlib import pyplot as plt

### Config

L = 5e-6 # Length of the beam [m]
w = t = 200e-9 # Width & Thickness for the rectangular beam [m]
rho = 2300 # Mass density for silicon [kg/m^3]
E = 1.85e11 # Young's modulus for silicon [Pa]

m = 1e-18

dx = 0.1e-6 # The size of the 1D mesh
k = 3 # The number of modes to calculate

### Script (Part (c))

N = math.floor(L / dx)
A = w*t
D = E*(w*(math.pow(t, 3)))/12

x = np.linspace(0, L, num=N)

# Construct a 4-th derivative operator with 2 fictitious nodes on each side
Dx4 = cd_1d_matrix(4, N + 4, dx)

x_m_vals = np.arange(1, 50)
x_m_vals = x_m_vals*0.1e-6

eigs = np.zeros_like(x_m_vals)
for xid in range(len(x_m_vals)):
    x_m = x_m_vals[xid]

    # Build mass perturbation
    p = np.ones(N+4, dtype=np.double)
    p = p/rho
    i_m = int(math.floor(x_m / dx)) + 2 # 2 Fictitious nodes at left boundary
    p[i_m] = p[i_m+1] = 1/(rho + m/(2*dx*A))
    p = sparse.diags(p, format="csr", dtype=np.double)

    B = p.dot(Dx4)

    # Apply 0th & 1st order homogenous BCs on the left boundary
    B = apply_1d_homogenous_bcs(B, [0, 1], 0)
    # Apply 2nd & 3rd order homogenous BCs on the right boundary
    B = apply_1d_homogenous_bcs(B, [2, 3], 1)

    # Calculate eigenvalues
    [e, v] = linalg.eigs(B, k=k, which="SM")  # which="SM" ensures that the low-frequency modes are selected.

    e = np.sort(e)

    eigs[xid] = e[0] * D / A

# Turn eigenvalues into frequencies
f_m = (0.5/math.pi) * np.sqrt(np.real(eigs))

### Part (d)

# Just use the fourth order derivative from earlier
# Apply 0th & 1st order derivative homogenous BCs on the left boundary
Dx4 = apply_1d_homogenous_bcs(Dx4, [1, 0], 0)

# Apply 2nd & 3rd order derivative homogenous BCs on the right boundary
Dx4 = apply_1d_homogenous_bcs(Dx4, [2, 3], 1)

# Find the low-frequency eigenmodes
[w2d_A, y] = linalg.eigs(Dx4/rho, k=k, which="SM") # which="SM" ensures that the low-frequency modes are selected.
w2d_A = np.real(w2d_A)
w2d_A = (0.5/math.pi) * np.sqrt(w2d_A / A * D ) # Calculate frequencies

y = L * y[:, 0] / (np.sum(np.real(y[:, 0])) * dx)
f_m_analytic = np.sqrt((w2d_A[0]*w2d_A[0]) / (1 + m/(rho*A*L)*y*y))

### Plot the results
plt.plot(x_m_vals, f_m/1e6, label="Numerical Solution")
plt.plot(x[:], f_m_analytic[:]/1e6, label="Analytic Solution")

plt.title("Lowest resonant frequency vs. mass position")
plt.xlabel("x_m [m]")
plt.ylabel("f [MHz]")
plt.legend()
plt.show()

