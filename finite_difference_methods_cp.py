import cupy as cp
import finite_difference_methods
from finite_difference_methods import *

def cd_1d_matrix_ND_v2(n, dim, domain):
    return cp._cupyx.scipy.sparse.csr_matrix(
        finite_difference_methods.cd_1d_matrix_ND_v2(n, dim, domain))

# Apply first-order radiating boundary conditions to the vector u on a given domain at the boundary bid
def apply_radiating_BC(u, v, dim, bid, n_cdt, domain):
    # u is the solution at the current timestep
    # v is the solution at the previous timestep
    # bid is a boundary id, 0 for lower and 1 for upper
    # n_cdt is the "velocity factor" encountered in the expressions for first order radiating boundary conditions
    # domain is the domain dict.

    h = domain['h'][dim].item()
    k = (domain['shape'][dim].item() - 1) * bid

    # To avoid editing the original matrices
    u = cp.copy(u)
    v = cp.copy(v)

    # Permute the input matrices for easier indexing
    if dim:
        u = cp.swapaxes(u, 0, dim)
        v = cp.swapaxes(v, 0, dim)

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
        u = cp.swapaxes(u, 0, dim)
        v = cp.swapaxes(v, 0, dim)

    return cp.copy(u)
