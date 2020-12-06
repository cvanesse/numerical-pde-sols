# Tests the pydistmesh package to see if it works identically to the matlab version

# Test the uniform mesh on a unit circle
import distmesh as dm
import numpy as np
from matplotlib import pyplot as plt

# Circle with equal mesh
fd = lambda p: np.sqrt((p**2).sum(1))-1
fd = lambda p: dm.dcircle(p, 0, 0, 1)
p, t = dm.distmesh2d(fd, dm.huniform, 0.1, (-1, -1, 1, 1))

plt.scatter(p[:, 0], p[:, 1])
plt.title("Circle with mesh")
plt.show()

# Rectangle with circular hole
def fd(p):
    return dm.ddiff(dm.drectangle(p, -1, 1, -1, 1),
                    dm.dcircle(p, 0, 0, 0.5))

def fh(p):
    return 0.05+0.3*dm.dcircle(p, 0, 0, 0.5)

p, t = dm.distmesh2d(fd, fh, 0.05, [-1, -1, 1, 1],
                     [(-1, -1), (-1, 1), (1, -1), (1, 1)])

plt.scatter(p[:, 0], p[:, 1])
plt.title("Rect with hole")
plt.show()
exit()