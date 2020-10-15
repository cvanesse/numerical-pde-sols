from finite_difference_methods import *
from matplotlib import pyplot as plt

domain = {
    "shape": [50,51],
    "size": 50*51,
    "h": [1, 1]
}

laplacian = cd_1d_matrix_ND_v2(2, 0, domain) + cd_1d_matrix_ND_v2(2, 1, domain)

lamb, V = linalg.eigsh(laplacian, k=5, which="SM")

V = V[:, 2].reshape(domain['shape'], order="F")

Y = np.arange(domain['shape'][0])*domain['h'][0]
X = np.arange(domain['shape'][1])*domain['h'][1]

X, Y = np.meshgrid(X, Y)
fig = plt.figure()
ax = plt.axes()
ax.contourf(X, Y, V, 100)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

#M = 2*sparse.eye(domain['size']) + 0.2 * laplacian

#print(M.toarray())

D2x = cd_1d_matrix_ND_v2(2, 0, domain)
D2y = cd_1d_matrix_ND_v2(2, 1, domain)
#D2z = cd_1d_matrix_ND_v2(2, 2, domain)
#print(D2x.toarray())
#print(D2y.toarray())
#print((D2y + D2x).toarray())