import numpy as np

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

    return ri + (num/den)

A = np.array([[0,0], [1.045, -2.4], [3.7, 1.6]])
print(find_circumcenter(A))