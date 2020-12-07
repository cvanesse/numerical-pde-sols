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


# Applies the 2D gradient of a function using linear lagrange basis function
def FE_gradient(F, P, T):
    # F is the function for which the gradient should be taken
    # P is the list of points in the domain
    # T is the list of triangles in the domain
    # returns grad: the gradient of F at the circumcenter

    Ne = np.shape(T)[0]
    grad = np.zeros((Ne, 2))

    for e in range(Ne):
        pts = P[T[e, :], :]

        A = np.ones((3, 3)); A[:, 1:] = pts
        A = 0.5 * np.linalg.det(A)  # The area of this triangle

        ab = compute_lagrange_coeff(pts)
        uq = np.transpose(np.vstack((F[T[e, :]], F[T[e, :]])))

        grad[e, :] = np.sum(ab*uq, axis=0) / (A)

    return grad



# Applies RBCs for a single edge of a single triangle
def apply_RBC_to_triangle(K, P, rc, i, j, eps_p):
    # K is the global stiffness matrix
    # b is the global forcing function
    # P is the list of points in the domain
    # rc is the location of the circumcenter
    # k is the index of the internal node
    # i is the index of the first (ordered CCW) boundary node
    # j is the index of the second (ordered CCW) boundary node
    # eps_p is the list of relative permittivities for all nodes in the domain.

    ri = P[i]  # First boundary node
    rj = P[j]  # Second boundary node

    eps_b = 0.5 * (eps_p[i] + eps_p[j])  # Relative permittivity of the boundary

    # Calculate geometric quantities for RBC application
    rb = 0.5 * (rj + ri)  # Radial vector pointing to the boundary
    mrb = np.linalg.norm(rb)
    urb = rb / np.linalg.norm(rb)  # Unit radial vector of the boundary

    n = 0.5*(rj-ri) - (rc-ri)
    un = n / np.linalg.norm(n) # Unit normal vector

    urbn = np.dot(urb, un)  # urb dot un [Normal component of urb]
    urbt = np.linalg.norm(urb - urbn * un)  # urb dot ut [Tangential component of urb]

    l = np.linalg.norm(ri - rj)  # Length of the boundary

    aburbn = eps_b / urbn
    lb6rblnrb = l / (6 * mrb * math.log(mrb))

    dKp = aburbn * (lb6rblnrb + urbt / 2)
    dKm = aburbn * (lb6rblnrb - urbt / 2)

    K[i, j] += dKp
    K[j, j] += dKp
    K[j, i] += dKm
    K[i, i] += dKm


# Applies radiating boundary conditions for finite-element poisson
def apply_RBCs(K, b, P, T, C, ids, ids_corner, eps_p):
    # K is the original stiffness matrix
    # b is the original forcing function
    # P is the set of points in the domain
    # T is the list of simplexes
    # C is the list of circumcenters corresponding to each simplex
    # ids is the list of node IDs on the boundary for which RBCs should be applied
    # eps_p is the list of permittivities evaluated at each vertex

    N_e = np.shape(T)[0]
    ids_all = np.hstack((ids, ids_corner))

    # Loop through each triangle, check if the triangle is "on the boundary"
    #   If it is, perturb the stiffness matrix according to the RBCs.
    for e in range(N_e):
        pts = T[e, :] # Global indices of the points.

        # Check which points (if any) are on the boundary
        bpts = [] # For edge and corner points
        opts = [] # For internal points
        for p in pts:
            if p in ids_all:
                bpts.append(p) # bpts will be in CCW order.
            else:
                opts.append(p)

        # Go to the next triangle if this one isn't on the boundary
        on_bdry = len(bpts) > 1
        if not on_bdry:
            continue

        # There are two possibilities (for a rectangle):
        #   either we have an internal point or a corner point
        if len(opts) == 1:
            # Two nodes are on the boundary, one internal point
            i = bpts[0]; j = bpts[1]
            #k = opts[0] but we don't care.

            apply_RBC_to_triangle(K, P, C[e], i, j, eps_p)
        else:
            # We have one corner point
            edges = [] # List of list for storing the local indices of edges in the correct order (CCW)
            for p in range(len(bpts)): # Find local index of the corner point
                if bpts[p] in ids_corner:
                    pn = (p+1)%3 # The next local node index (in CCW order)
                    p2n = (p+2)%3 # The next next local node index (in CCW order0
                    edges.append([bpts[p], bpts[pn]]) # Add the edge CCW to the corner
                    edges.append([bpts[p2n], bpts[p]]) # Add the edge CW to the corner (in the correct order)
                    break

            # Apply the RBCs for both edges
            for edge in edges:
                apply_RBC_to_triangle(K, P, C[e], edge[0], edge[1], eps_p)

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
