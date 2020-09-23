from finite_difference_methods import *
from matplotlib import pyplot as plt

### Config

# Physics
hbar = 1.0546e-34
me = 9.109e-31
q=160.218e-21

# Problem Definition
a = 1e-9
d = 5e-9
V0 = 1 # Boundary height [eV]
Vb = 0 # Bias voltage [V]

# Simulation config
dx = 1e-11 # Grid size [m]
k = 2 # Number of eigenmodes to calculate

### Code
x = np.linspace(-d, d, int(d/dx))

## Construct the Hamiltonian
H = cd_1d_matrix(2, len(x)+2, dx) # 2nd order derivative, 2 fictitious nodes
H = (-hbar*hbar/2/me) * H

V = V0*q*np.ones(len(x)+2)

# Fill box with linear distribution representing the bias perturbation
box_id_left = np.argmin(np.abs(x-(-a/2)))+1
box_id_right = np.argmin(np.abs(x-(a/2)))+1
V[box_id_left:box_id_right] = np.linspace(0, Vb*q, box_id_right-box_id_left)

# Set right boundary higher according to the bias perturbation
V[box_id_right:] = V[box_id_right:] + Vb*q

# Add the bias on the diagonals of the hamiltonian
H = H + sparse.diags(V)

# Add boundary conditions to the hamiltonian
H = apply_1d_homogenous_bcs(H, [1], 0)
H = apply_1d_homogenous_bcs(H, [1], 1)

## Find the eigenvalues of the hamiltonian
[E, psi] = linalg.eigs(H, k=k, which="SM")
E = np.real(E)

for mid in range(k):
    plt.plot(x, np.power(np.abs(psi[:, mid]), 2), label=("E = %.2f eV" % (E[mid]/q)))

plt.title("First %d States, Vb = %.2f" % (k, Vb))
plt.xlabel("x [m]")
plt.ylabel("|psi|^2(x) [m]")
plt.legend()
plt.show()