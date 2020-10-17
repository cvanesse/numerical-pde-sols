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

# Builds a square block-diagonal matrix using a list of blocks, diagonal offset, and the number of diagonal blocks N
def block_diag(blocks, diags, N, format="csr"):
    # Verify proper inputs
    n = np.shape(blocks[0])
    if (len(n) != 2) :
        print("ERROR: Invalid block dimension for block_diag! Must be 2D")
        exit()
    for b in blocks:
        m = np.shape(b)
        if m[0] != m[1]:
            print("ERROR: Invalid block shape for block_diag! Must be square matrices")
            exit()
        if m[0] != n[0]:
            print("ERROR: Invalid block list for block_diag! Must be a list of square matrices of the same size.")
            exit()

    # Build block-diagonal matrix using provided blocks and kronecker products.
    M = sparse.csr_matrix((n[0]*N, n[0]*N))
    for did in range(len(diags)):
        m = sparse.diags([1], diags[did], (N, N), format=format)
        m = sparse.kron(m, blocks[did], format=format)
        M = m+M

    return M

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

# Constructs an nth-order central-difference derivative operator matrix for a domain of arbitrary dimensionality
def cd_1d_matrix_ND_v2(n, dim, domain):
    # n is the order of the derivative
    # dim is the dimension of the derivative [starting from 0]
    # domain is a dict containing:
    # domain["shape"]: integer containing the number of nodes for each dimension in the domain
    # domain["size"]: integer containing the number of nodes in the domain
    # domain["h"]: vector containing the real-space discretization size for each dimension

    h = domain["h"][dim]

    # First, construct the identity matrix of one dimension lower than the derivative of interest
    if dim == 0:
        eye = sparse.csr_matrix([1])
    else:
        eye = sparse.eye(np.round(np.prod(domain["shape"][:dim])), format="csr")

    # Then, use the computational molecule for the derivative of interest to construct the blocks for this dimension
    mol = cd_1d_molecule(n)
    blocks = [mol[i]*eye for i in range(len(mol))]

    idx = np.arange(len(blocks)) - math.floor(len(blocks) / 2)  # Index the entries of the computational molecule

    # Build the operator for the dimension of interest
    M = block_diag(blocks, idx, domain["shape"][dim])

    # Build the operator for the entire domain, by using the previous operator as diagonal blocks (if necessary)
    if (dim+1 < len(domain["shape"])):
        M = block_diag([M], [0], np.round(np.prod(domain["shape"][dim+1:])))

    # Return the result.
    return M / math.pow(h, n)

# Gets the 1D node number from the position and domain
def get_node_number(pos, domain):
    pos = np.array([pos[-i] for i in range(len(pos))])
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
    k = (domain['shape'][dim] - 1) * bid

    # To avoid editing the original matrices
    u = np.copy(u)
    v = np.copy(v)

    # Permute the input matrices for easier indexing
    if dim:
        u = np.swapaxes(u, 0, dim)
        v = np.swapaxes(v, 0, dim)

    # The internal nodes are located at k+1 when the left (bid=0) boundary is used
    # Otherwise they are located at k-1
    shift = (-1)**(bid)

    # Build the coefficients of the general first-order RBCs
    A = (n_cdt + (1 / h))
    B = (-n_cdt + (1 / h))

    # Indexing an N-dimensional matrix u with a single index will take the cross section
    #   At that point along the first dimension.
    u[k] = (1/A)*(B*u[k+shift] - B*v[k] + A*v[k+shift])

    if dim:
        u = np.swapaxes(u, 0, dim)
        v = np.swapaxes(v, 0, dim)

    return np.copy(u)
