import os
import torch
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from PySEC import generate_data as gd
from PySEC.nystrom_cidm import cidm, nystrom #, nystrom_grad
from PySEC.del1 import del1_as_einsum
from PySEC.sec_utils import reshape_fortran

N = 1000
torch.manual_seed(1)
dtype = torch.float64

t = torch.linspace(0, 2*torch.pi, N, dtype=dtype)
theta = 8 * t
phi = t
r, R = 1.0, 2
rct = r * torch.cos(theta)
x, y, z = (R + rct) * torch.cos(phi), (R + rct) * torch.sin(phi), r * torch.sin(theta)
data = torch.stack([x, y ,z]).t()
data += torch.randn(data.shape) * 6.0e-2

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.view_init(33, 55)
ax.set_box_aspect(np.ptp(data, axis=0))
ax.scatter(*data.t(), c=t, cmap='hsv', s=4)
ax.set_xlim(-1.4 * R, 1.4 * R), ax.set_ylim(-1.4 * R, 1.4 * R)
ax.set_zlim(-1.1, 1.1), ax.set_zticks([-1, 0, 1])
plt.show(), plt.close(fig)

nvar=200
k=200
k2=14

u0, l0, peq0, qest0, _, _, KP = cidm(data, k=k, k2=k2, nvars=nvar)

fig, ax = plt.subplots(figsize=(4,4))
# ax.scatter(u0[:, 1] / l0[1], u0[:, 2] / l0[2], c=t, cmap='hsv', s=4)
ax.scatter(u0[:, 1], u0[:, 2], c=t, cmap='hsv', s=4)
ax.set_aspect('equal')
ax.set_xticks([]), ax.set_yticks([])
ax.set_xlabel(r'$\phi_1$'), ax.set_ylabel(r'$\phi_2$')
plt.show(), plt.close(fig)

Xhat = u0[:, :12].t() @ (data * peq0[:, None])
xrecon = u0[:, :12] @ Xhat

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.view_init(33, 55)
ax.set_box_aspect(np.ptp(data, axis=0))
ax.set_xlim(-1.4 * R, 1.4 * R), ax.set_ylim(-1.4 * R, 1.4 * R)
ax.set_zlim(-1.1, 1.1), ax.set_zticks([-1, 0, 1])
ax.scatter(*xrecon.t(), c=t, cmap='hsv', s=4)
plt.show(), plt.close(fig)

Xhat = u0[:, :].t() @ (data * peq0[:, None])
xrecon = u0[:, :] @ Xhat

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.view_init(33, 55)
ax.set_box_aspect(np.ptp(data, axis=0))
ax.set_xlim(-1.4 * R, 1.4 * R), ax.set_ylim(-1.4 * R, 1.4 * R)
ax.set_zlim(-1.1, 1.1), ax.set_zticks([-1, 0, 1])
ax.scatter(*xrecon.t(), c=t, cmap='hsv', s=4)
fig.tight_layout()
plt.show(), plt.close(fig)



debug_var = 1
