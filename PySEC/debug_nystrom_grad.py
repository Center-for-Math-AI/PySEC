import torch
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from PySEC import generate_data as gd
from PySEC.nystrom_cidm import cidm, nystrom, nystrom_grad
from PySEC.del1 import del1_as_einsum
from PySEC.sec_utils import reshape_fortran


test = gd.generate_circle(100)
points = test[0].t()
# def cidm(x, nvars=None, k=None, k2=None, tuning_method=None, **knn_args):
u0, l0, peq0, qest0, eps0, dim0, KP = cidm(points)
Xhat = u0.T @ torch.diag(peq0) @ \
            points.view(points.shape[0], -1).to(dtype=u0.dtype, device=u0.device)

cidm_ss = slice(1, 4)
xhat2 = u0.T @ torch.diag(peq0) @ u0[:, cidm_ss]
xhat3 = u0.T @ torch.diag(peq0) @ u0[:, :]

n1 = min(80, KP.k)  # use kNN size capped at 80
# n1 = l.shape[0] - 10
u1, l1, d1, _, h1, _ = del1_as_einsum(u0, l0, torch.diag(peq0), n1)

u2, peq2, qest2, gradu = nystrom_grad(points, KP)


debug_var = 1