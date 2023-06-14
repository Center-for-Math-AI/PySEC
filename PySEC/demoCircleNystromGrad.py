import os
import torch
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from PySEC import generate_data as gd
from PySEC.nystrom_cidm import cidm, nystrom, nystrom_grad, nystrom_gradu
from PySEC.del1 import del1_as_einsum
from PySEC.sec_utils import reshape_fortran


# test = gd.generate_circle(100)
test = gd.generate_noisy_circle(180, 1e-2)
points = test[0].t()
t = test[1]
angs = torch.atan2(t[1], t[0]) % (2 * torch.pi)
ss = slice(None, None, 6)
xsub = points[ss, :]
tsub = angs[ss]



# def cidm(x, nvars=None, k=None, k2=None, tuning_method=None, **knn_args):
k = 40
k2 = 10
u0, l0, peq0, qest0, eps0, dim0, KP = cidm(points, k=k, k2=k2, nvars=4*k)
Xhat = u0.T @ torch.diag(peq0) @ points.view(points.shape[0], -1).to(u0)
Shat = u0.T @ torch.diag(peq0) @ angs.view(-1, 1)

xrecon = u0 @ Xhat
arecon = u0 @ Shat


fig, ax = plt.subplots(figsize=(4,4))
# ax.plot(angs - angs, label='intrinsic')
for i in [40, 80, 120]:
    Shat0 = u0[:,:i].t() @ torch.diag(peq0) @ angs.view(-1, 1)
    arecon = u0[:, :i] @ Shat0
    ax.plot(torch.rad2deg(arecon[:, 0] - angs), label=f'$U_0(0:{i})$')

ax.set_xlabel('data position (degrees)')
ax.set_ylabel('reconstruction error (degrees)')
ticks = list(range(0, 181, 30))
tick_labels = [f'{2*i}' for i in ticks]
ax.set_xticks(ticks, tick_labels)
# ax.set_aspect('equal')
ax.legend()
fig.tight_layout()
plt.show()
debug_var = 1


fig, aax = plt.subplots(1, 2, figsize=(9,4))
for i in [40, 80, 120]:
    Ihat0 = u0[:, :i].t() @ torch.diag(peq0) @ t.t()
    irecon = u0[:, :i] @ Ihat0
    aax[0].plot(torch.rad2deg(irecon[:, 0] - t[0]), label=f'$U_0(0:{i})$')
    aax[1].plot(torch.rad2deg(irecon[:, 1] - t[1]), label=f'$U_0(0:{i})$')


ticks = list(range(0, 181, 30))
tick_labels = [f'{2*i}' for i in ticks]
aax[0].set_xticks(ticks, tick_labels)
aax[1].set_xticks(ticks, tick_labels)
aax[0].set_xlabel('data position (degrees)')
aax[0].set_ylabel('reconstruction error (degrees)')
aax[1].set_xlabel('data position (degrees)')
aax[1].set_ylabel('reconstruction error (degrees)')
# ax.set_aspect('equal')
aax[0].legend()
fig.tight_layout()
plt.show()
debug_var = 1


u2, peq2, qest2, gradu = nystrom_grad(xsub, KP)


fig, aax = plt.subplots(2, 2, figsize=(6, 6))
for ii, ax in enumerate(aax.flat):
    ax.plot(xsub[:,0], xsub[:,1], markersize=4, marker='.', linestyle='')
    gradui = gradu[:, ii + 1]
    ax.quiver(xsub[:, 0], xsub[:, 1], gradui[:, 0], gradui[:, 1], color='tab:red')
    ax.set_aspect('equal')
    ticks = torch.arange(-1, 2, 1)
    ax.set_xticks(ticks), ax.set_yticks(ticks)
    ax.set_title(r'$\nabla U_{0}($' + f'{ii}' + r'$)$')

fig.tight_layout()
plt.show(), plt.close(fig)


debug_var = 1


u22, peq22, qest22, gradu2 = nystrom_gradu(xsub, KP)

fig, ax = plt.subplots()
ax.scatter(*u2[:, 1:3].t().cpu())
gradu2i = gradu2[:, 1, :]
ax.quiver(*u2[:, 1:3].t().cpu(), *gradu2i[:, 1:3].t().cpu(), color='tab:red')
plt.show()
