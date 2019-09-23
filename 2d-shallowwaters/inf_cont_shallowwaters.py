import meshio
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


print("At t=0")
eqnPath = "2d-shallowwaters/"
mesh = meshio.read(os.path.join(eqnPath, "data", "0FV-Paraview 0s .vtk"))
print("Mesh points")
print(mesh.points.shape)
print("Mesh values")
for key in mesh.point_data:
    print(f"{key}: {mesh.point_data[key].shape}")


N_u = 1000
idx = np.random.choice(mesh.points.shape[0], N_u, replace=False)
X_u = mesh.points[idx, 0:3]
u = mesh.point_data["h"][:, None][idx, :]

print("X_u:", X_u.shape)
print("u:", u.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_u[:, 0:1], X_u[:, 1:2], X_u[:, 2:3])
#ax.scatter(X_u[:, 0:1], X_u[:, 1:2], u)
plt.show()
