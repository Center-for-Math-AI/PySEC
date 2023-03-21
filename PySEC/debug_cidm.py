import torch
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from PySEC import generate_data as gd
from PySEC.nystrom_cidm import cidm
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
u1, l1, d1, _, h1, _ = del1_as_einsum(
    u0, l0, torch.diag(peq0), n1
)

v1, v2 = 30, 120  # 3d view angles
print(f'view at ({v1},{v2})')  # 91, 44 is top down for circle
ss = slice(None, None, 4)  # step size to subset data
vec_count = 9
nrows = 3
ncols = ceil(vec_count / nrows)
x_3d = u0[:, cidm_ss]
fig = plt.figure(figsize=(ncols * 4, nrows * 4))  # subplots(1, 1)
for iu in range(vec_count):
    umatrix = reshape_fortran(h1.T @ u1[:, iu], [n1, n1])  # reshape_fortran necessary here
    datavectorfield = u0[:, :n1] @ umatrix @ xhat2[:n1, :]
    # replace u0_sub with u from nystrom
    # dvec_3d = mgb.to_svd(datavectorfield, 4, order='c')
    dvec_3d = datavectorfield
    dvec_3d = dvec_3d[:, :]  # ignore constant svd coeff

    ax = fig.add_subplot(nrows, ncols, iu + 1, projection='3d')
    ax.scatter(x_3d[:, 0], x_3d[:, 1], x_3d[:, 2], alpha=0.8, s=0.8)
    ax.quiver(x_3d[ss, 0], x_3d[ss, 1], x_3d[ss, 2],
              dvec_3d[ss, 0], dvec_3d[ss, 1], dvec_3d[ss, 2],
              length=300.0, normalize=False, color='red',
              arrow_length_ratio=0.02)
    ax.view_init(v1, v2)
    ax.set_title(f'SEC U({iu})')

fig_title = f'Del1 vector field (SVD basis) for circle'
fig.suptitle(fig_title, y=0.99)
fig.tight_layout(h_pad=2)
plt.show()
plt.close(fig)


u0.shape

debug_var = 1