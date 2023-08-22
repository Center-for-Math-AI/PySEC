import os
import torch
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from PySEC import generate_data as gd
from PySEC.nystrom_cidm import cidm, nystrom, nystrom_grad
from PySEC.del1 import del1_as_einsum
from PySEC.sec_utils import reshape_fortran


# test = gd.generate_circle(100)
test = gd.generate_noisy_circle(120, 1e-6)
points = test[0].t()
t = test[1]
angs = torch.atan2(t[1], t[0])

# def cidm(x, nvars=None, k=None, k2=None, tuning_method=None, **knn_args):
u0, l0, peq0, qest0, eps0, dim0, KP = cidm(points, k=40, k2=10)
Xhat = u0.T @ torch.diag(peq0) @ points.view(points.shape[0], -1).to(u0)

n1 = min(80, KP.k)  # use kNN size capped at 80
u0_sub = u0[:, :n1]
xhat0 = u0_sub.T @ torch.diag(peq0) @ points.view(points.shape[0], -1).to(u0_sub)
u1, l1, d1, _, h1, _ = del1_as_einsum(u0, l0, torch.diag(peq0), n1)

# n1 = l.shape[0] - 10
# xhat = points @ peq0 * u0[:, :n1] #; % % % Fourier transform of embedding
# ihat = intrinsic * D * u(:, 1: n1); % % % Fourier

fig, aax = plt.subplots(2, 2, figsize=(6,6))
for iu in range(4): #= 1:4 # % % % Convert j - th eigenfield U(:, j) from frame representation

    umatrix = reshape_fortran(h1.T @ u1[:, iu], [n1, n1])  # reshape_fortran necessary here
    datavectorfield = u0_sub @ umatrix @ xhat0

    ax = aax.flat[iu]
    ss = slice(None, None, 6)
    xp = points[ss, :2].t()
    tans = datavectorfield[ss, :2].t()
    ax.plot(*xp, markersize=4, marker='.', linestyle='')
    ax.quiver(*xp, *tans, color='tab:red')
    # quiver(intrinsic(1,:), intrinsic(2,:), intrinsicvectorfield(1,:), intrinsicvectorfield(2,:), 1, 'r');
    ax.set_aspect('equal')
    ticks = torch.arange(-1, 2, 1)
    ax.set_xticks(ticks), ax.set_yticks(ticks)
    ax.set_title(f'$U_1({iu})$')

fig.tight_layout()
plt.show()

debug_var = 1
