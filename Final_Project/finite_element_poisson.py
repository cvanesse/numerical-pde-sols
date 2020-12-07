# Helper functions for finite element analysis of the poisson equation
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import math


def extract_submatrix(M, i_vals, j_vals):
    M_out = M[i_vals,]
    M_out = M_out[:, j_vals]
    return M_out


def zero_submatrix(M, i_vals, j_vals):
    #print("Zeroing submatrix")
    for i in i_vals:
        M[i, j_vals] *= 0
    return M


def set_submatrix(M, Mp, i_vals, j_vals):
    #print("Setting submatrix")
    for i in range(len(i_vals)):
        M[i_vals[i], j_vals] = Mp[i, :]
    return M


def apply_dirichlet_conditions(K, b, ids, ids_domain, vals):
    K = K.tocsr()

    # Extract the interaction matrix between the boundary nodes and the domain
    M_cent_bound = extract_submatrix(K, ids_domain, ids)

    b[ids_domain] -= M_cent_bound.dot(vals)
    b[ids] = vals
    K = zero_submatrix(K, ids, ids_domain)
    K = zero_submatrix(K, ids_domain, ids)
    K = set_submatrix(K, sparse.eye(np.size(ids), format="lil"), ids, ids)

    K.eliminate_zeros()

    return K, b


# Computes the coefficients of the lagrange basis functions
def compute_lagrange_coeff(pts):
    idx = np.array([[1, 2], [2, 0], [0, 1]])

    ab = np.zeros((3, 2))
    for i in range(3):
        ab[i, 0] = pts[idx[i, 0], 1] - pts[idx[i, 1], 1]
        ab[i, 1] = pts[idx[i, 1], 0] - pts[idx[i, 0], 0]

    return ab


def compute_local_eq(pts, eps_e, eps_p):
    # Inputs:
    #   pts: A [3, 2] matrix containing the vertices of the triangle
    #   eps_e: A value of the relative permittivity at the circumcenter
    #   eps_p: A [3, 1] matrix containing the permittivity at each point in the triangle

    # Explicitly calculate coefficients for

    A = np.ones((3, 3)); A[:, 1:] = pts
    A = 0.5*np.linalg.det(A) # The area of this triangle
    eps_o_fA = eps_e / 4*(A)

    ab = compute_lagrange_coeff(pts)

    Ke = np.zeros((3, 3))
    be = np.zeros((3)) # Forcing function is 0 for all problems (sourceless)
    for p in range(3):
        ap = ab[p, 0]
        bp = ab[p, 1]
        for q in range(3):
            aq = ab[q, 0]
            bq = ab[q, 1]
            Ke[p, q] = -(ap*aq + bp*bq) * eps_o_fA

    return Ke, be


def construct_poisson_eq(P, T, eps_e, eps_p):
    # Inputs:
    #   P - The points in the computational domain
    #   T - The simplices of the computational domain (from Delaunay (P))
    #   C - The circumcenters of each simplex
    #   eps_e - The relative permittivity at the circumcenter of each simplex
    #   eps_p - The relative permittivity at each point

    N_e = np.shape(T)[0]
    N_p = np.shape(P)[0]

    # Algorithm Credit: Dr. Vien Van, University of Alberta, 2020
    K = sparse.lil_matrix((N_p, N_p))
    b = np.zeros(N_p)

    for e in range(N_e):
        ee = eps_e[e]
        pts = P[T[e, :], :]
        ep = eps_p[T[e, :]]

        Ke, be = compute_local_eq(pts, ee, ep)

        for p in range(3):
            for q in range(3):
                K[T[e, p], T[e, q]] += Ke[p, q]
            b[T[e, p]] += be[p]

    return K, b


# Calculates the euclidean distance between two vectors
def dist(v1, v2):
    dv = v1-v2
    return math.sqrt(dv[0]**2 + dv[1]**2)


# Finds the circumcenter of the triangle defined by coordinates of the rows of the triangle
def find_circumcenter(tri):
    zeros = np.zeros((3, 1))
    tri = np.concatenate((tri, zeros), axis=1)
    ri = tri[0, :]

    dr = [tri[i+1, :]-ri for i in range(2)]
    mdr2 = [np.sum(np.power(d, 2)) for d in dr]

    dmdr2dr = mdr2[1]*dr[0] - mdr2[0]*dr[1]

    drxdr = np.cross(dr[1], dr[0])

    num = np.cross(dmdr2dr, drxdr)
    den = 2*np.sum(np.power(drxdr, 2))

    C = ri + (num/den)

    return C[:2]


def calculate_circumcenters(P, T):
    # Calculates all of the circumcenters and radii for the entire computational domain
    print("Calculating circumcenters for entire domain...")

    N = np.shape(T)[0]

    centers = np.zeros((N, 2))
    for e in range(N):
        p = P[T[e], :]
        centers[e] = find_circumcenter(p)

    return centers
