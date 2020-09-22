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

# Constructs a finite-difference nth order derivative operator matrix with size NxN
def cd_1d_matrix(n, N, h):
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
        # Right
        A_f = A[num_fictitious:2*num_fictitious, :num_fictitious]
        A = A[num_fictitious:, num_fictitious:]
        F = cd_1d_homogenous_bc_matrix(orders, side)
        dA = np.matmul(A_f.toarray(),F)

        A[:dA.shape[0], :dA.shape[1]] = A[:dA.shape[0], :dA.shape[1]] + dA

    return A



