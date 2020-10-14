from finite_difference_methods import *

domain = {
    "shape": [2, 2, 2],
    "size": [8],
    "h": [1, 1, 1]
}

D2x = cd_1d_matrix_ND_v2(2, 0, domain)
D2y = cd_1d_matrix_ND_v2(2, 1, domain)
D2z = cd_1d_matrix_ND_v2(2, 2, domain)