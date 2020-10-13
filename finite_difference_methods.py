from scipy import sparse
from scipy.sparse import linalg
import numpy as np
import math

# Constructs the lowest-order accurate computational molecule for an nth order central-difference derivative
def cd_1d_molecule(n):
    n_even = n-n%2
    mol = np.array([1])
    for o in range(n_even):
        mol_R = np.append(np.array([0]), mol)
        mol_L = np.append(mol, np.array([0]))
        mol = mol_R - mol_L

    if (n%2):
        mol_R = np.append(np.array([0,0]), mol)
        mol_L = np.append(mol, np.array([0,0]))
        mol = (mol_R - mol_L)/2

    return mol

# Constructs a finite-difference nth order central-difference derivative operator matrix with size NxN
def cd_1d_matrix(n, N, h):
    if (n == 0):
        sparse.diags([1], [0], shape=(N, N), format="csr")
    mol = cd_1d_molecule(n)
    idx = np.arange(len(mol)) - math.floor(len(mol)/2)
    return sparse.diags(mol, idx, shape=(N, N), format="csr") / math.pow(h, n)

# Generates the homogenous BC perturbation matrix from a list of derivative orders
def cd_1d_homogenous_bc_matrix(orders, side):
    # Get molecules
    mols = []
    max_size = -1
    for n in range(len(orders)):
        mol = cd_1d_molecule(orders[n])
        max_size = max([max_size, len(mol)])
        mols.append(mol)

    midpoint = int(np.floor(max_size/2))
    beta = np.zeros((len(orders), max_size))

    for mid in range(len(mols)):
        n = math.floor(len(mols[mid])/2)
        idx = np.arange(-n, n+1) + midpoint
        beta[mid, idx] = mols[mid]

    if (side):
        beta_f = beta[:, -len(orders):]
        beta_c = beta[:, :-len(orders)]
    else:
        beta_f = beta[:, :len(orders)]
        beta_c = beta[:, len(orders):]
    beta_f = np.linalg.inv(beta_f)
    F = -np.matmul(beta_f, beta_c)

    return F

# Applies homogenous boundary conditions of the order desired at the side desired
def apply_1d_homogenous_bcs(A, orders, side):
    num_fictitious = len(orders)
    if side:
        # Right
        A_f = A[-2*num_fictitious:-num_fictitious, -num_fictitious:]
        A = A[:-num_fictitious, :-num_fictitious]
        F = cd_1d_homogenous_bc_matrix(orders, side)
        dA = np.matmul(A_f.toarray(),F)

        A[-dA.shape[0]:, -dA.shape[1]:] = A[-dA.shape[0]:, -dA.shape[1]:] + dA
    else:
        # Left
        A_f = A[num_fictitious:2*num_fictitious, :num_fictitious]
        A = A[num_fictitious:, num_fictitious:]
        F = cd_1d_homogenous_bc_matrix(orders, side)
        dA = np.matmul(A_f.toarray(),F)

        A[:dA.shape[0], :dA.shape[1]] = A[:dA.shape[0], :dA.shape[1]] + dA

    return A

# Constructs an nth-order central-difference derivative operator matrix for an arbitrary dimension domain.
def cd_1d_matrix_ND(n, dim, domain):
    # n is the order of the derivative
    # dim is the dimension of the derivative [starting from 0]
    # domain is a dict containing:
    # domain["shape"]: integer containing the number of nodes for each dimension in the domain
    # domain["size"]: integer containing the number of nodes in the domain
    # domain["h"]: vector containing the real-space discretization size for each dimension

    N = domain["size"]
    h = domain["h"][dim]

    if (n == 0):
        sparse.diags([1], [0], shape=(N, N), format="csr")

    mol = cd_1d_molecule(n) # Get the 1D central-difference computational molecule for derivative of order n

    idx = np.arange(len(mol)) - math.floor(len(mol)/2) # Index the entries of the computational molecule

    # Multiply the index by an appropriate number to reference higher indices.
    idx = idx * np.prod(domain["shape"][:dim])

    return sparse.diags(mol, idx, shape=(N, N), format="csr") / math.pow(h, n)

# Gets the 1D node number from the position and domain
def get_node_number(pos, domain):
    pos = np.array(pos)
    if (len(pos) != len(domain['h'])):
        print("ERROR: Mismatch between point dimensions and domain dimensions.")
    pos = np.round(pos / domain['h'])
    for i in range(len(pos)-1):
        pos[i+1:] = domain['shape'][i]*pos[i+1:]
    return np.sum(pos)

# Apply first-order radiating boundary conditions to the vector u on a given domain at the boundary bid
def apply_radiating_BC(u, v, dim, bid, n_cdt, domain):
    # u is the solution at the current timestep
    # v is the solution at the previous timestep
    # bid is a boundary id, 0 for lower and 1 for upper
    # n_cdt is the "velocity factor" encountered in the expressions for first order radiating boundary conditions
    # domain is the domain dict.

    h = domain['h'][dim]
    k = domain['shape'][dim] * bid - 1

    # Permute the input matrices for easier indexing
    u = np.swapaxes(u, 0, dim)
    v = np.swapaxes(v, 0, dim)

    # Second, build the coefficients of the general first-order RBCs
    A = ((-1.0) ** bid) * (n_cdt + (1 / h))
    B = ((-1.0) ** bid) * (-n_cdt + (1 / h))
    C = (((-1.0) ** bid) * n_cdt + (1 / h))
    D = (((-1.0) ** bid) * n_cdt - (1 / h))

    # The internal nodes are located at k+1 when the left (bid=0) boundary is used
    # Otherwise they are located at k-1
    shift = (-1)**(bid)

    # Indexing an N-dimensional matrix u with a single index will take the cross section
    #   At that point along the first dimension.
    u[k] = (1/A)*(B*u[k+shift] + C*v[k] + D*v[k+shift])

    u = np.swapaxes(u, 0, dim)
    v = np.swapaxes(v, 0, dim)
    return u
