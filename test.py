from finite_difference_methods import *

domain = {
    "shape": [2, 4],
    "size": 2*4,
    "h": [1, 1]
}

laplacian = cd_1d_matrix_ND_v2(2, 0, domain) + cd_1d_matrix_ND_v2(2, 1, domain)
M = 2*sparse.eye(domain['size']) + 0.2 * laplacian

print(M.toarray())

D2x = cd_1d_matrix_ND_v2(2, 0, domain)
D2y = cd_1d_matrix_ND_v2(2, 1, domain)
#D2z = cd_1d_matrix_ND_v2(2, 2, domain)
print(D2x.toarray())
print(D2y.toarray())
print((D2y + D2x).toarray())