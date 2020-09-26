from finite_difference_methods import *
from matplotlib import pyplot as plt

### Config

## Note: Ensuring that there is a mesh-boundary at the edge of the box
#        is VERY important. Without this erroneous eigenmodes are calculated

# Physics
hbar = 1.0546e-34 # hbar in J*s
me = 9.109e-31 # Electron mass in kg
q = 1.60218e-19 # Electron charge in C

# Problem Definition
a = 1e-9 # 0.5*Well size [m]
d = 5e-9 # 0.5*Simulation size [m]
V0 = 1 # Boundary height [eV]
Vb = 0.5 # Bias voltage [V]

# Simulation config
dx = 1e-11 # Grid size [m]
k = 2 # Number of eigenmodes to calculate

### Code
x = np.linspace(-d, d, int((2*d)/dx)+1)

## Construct the Hamiltonian
H = cd_1d_matrix(2, len(x)+2, dx) # 2nd order derivative, 2 fictitious nodes
H = (-hbar*hbar/(2*me)) * H

V0 = V0*q
Vb = Vb*q
V = V0*np.ones(len(x)+2)

# Fill box with linear distribution representing the bias perturbation
box_id_left = np.argmin(np.abs(x-(-a/2)))
box_id_right = np.argmin(np.abs(x-(a/2)))+1
V[box_id_left:box_id_right] = np.linspace(0, Vb, box_id_right-box_id_left)

# Set right boundary higher according to the bias perturbation
V[box_id_right:] = V[box_id_right:] + Vb


# Add the bias on the diagonals of the hamiltonian
H = H + sparse.diags(V)

# Add boundary conditions to the hamiltonian
H = apply_1d_homogenous_bcs(H, [1], 0) # Neumann (1st order derivative) on left
H = apply_1d_homogenous_bcs(H, [1], 1) # Neumann (1st order derivative) on right

#H = H/np.max(H.toarray())
#print(H)

## Find the eigenvalues of the hamiltonian
[E, psi] = linalg.eigs(H, k=k, which="SM")
E = np.real(E)

for mid in range(k):
    plt.plot(x, np.real(psi[:, mid]), label=("E = %.2f eV" % (E[mid]/q)))

plt.title("First %d States, Vb = %.2f" % (k, Vb/q))
plt.xlabel("x [m]")
plt.ylabel("|psi|^2(x) [m]")
plt.legend()
plt.show()