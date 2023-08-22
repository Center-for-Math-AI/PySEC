import os, time
import torch
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from PySEC import generate_data as gd
from PySEC.nystrom_cidm import cidm, nystrom #, nystrom_grad, nystrom_gradu
from PySEC.del1 import del1_as_einsum
from PySEC.sec_utils import reshape_fortran

torch.backends.cudnn.deterministic = True  # makes batch norm deterministic
torch.manual_seed(0)

# def divergence(y, x):
#     div = 0.
#     for i in range(y.shape[-1]):
#         div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
#     return div
# def divergence(f):
#     """
#     Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
#     :param f: List of ndarrays, where every item of the list is one dimension of the vector field
#     :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
#     """
#     num_dims = len(f)
#     return np.ufunc.reduce(np.add, [np.gradient(f[num_dims - i - 1], axis=i) for i in range(num_dims)])

class SEC():
    def __init__(self, data, intrinsic_params=None):
        self.data = data
        self.intrinsic_params = intrinsic_params

    def fit(self):
        t0 = time.time()
        ret0 = cidm(self.data, )  # k=k, k2=k2, nvars=4*k)
        t1 = time.time()
        print(f'Time for CIDM: {t1 - t0:.2f}s')
        self.u, self.l, self.peq, self.qest, self.eps, self.dim, self.KP = ret0

        self.Xhat = self.u.T @ torch.diag(self.peq) @ self.data.view(self.data.shape[0], -1).to(self.u)
        self.Shat = None
        if self.intrinsic_params is not None:
            self.Shat = self.u.T @ torch.diag(self.peq) @ self.intrinsic_params

        self.n1 = min(80, self.KP.k)  # use kNN size capped at 80
        t2 = time.time()
        ret1 = del1_as_einsum(self.u, self.l, torch.diag(self.peq), self.n1)
        t3 = time.time()
        print(f'Time for Del1: {t3 - t2:.2f}s')
        self.u1, self.l1, self.d1, _, self.h1, _ = ret1

        return self


# test = gd.generate_circle(100)
data, ip = gd.generate_torus(2000, noise=1e-2)
points = data.t() #test[0].t()
t = ip.t()
angs = torch.atan2(t[1], t[0]) % (2 * torch.pi)
ss = slice(None, None, 6)
xsub = points[ss, :]
tsub = angs[ss]


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(*data, c=ip[1], cmap='hsv')
ax.set_title('Az')
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(*data, c=ip[0], cmap='hsv')
ax.set_title('DLA')
plt.show(), plt.close(fig)

geom = SEC(points, t)
geom = geom.fit()
# xhat0 = u0_sub.T @ torch.diag(peq0) @ points.view(points.shape[0], -1).to(u0_sub)
gip = geom.intrinsic_params
jidx = geom.KP.X.shape[0] // 2

plot_vec_list = [
    [1, 2, 3],
    [1, 2, 4],
    [1, 2, 5],
    [3, 4, 1],
    # [2, 3, 5],
]
view_az, view_dl = 20, 40
fig = plt.figure(figsize=(8, 4 * len(plot_vec_list)))
for ii, plot_vecs in enumerate(plot_vec_list):
    ax = fig.add_subplot(len(plot_vec_list), 2, 1 + ii*2, projection='3d')
    ax.view_init(view_dl, view_az), ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
    ax.set_xlabel(f'U{plot_vecs[0]}', labelpad=-6), ax.set_ylabel(f'U{plot_vecs[1]}', labelpad=-6), ax.set_zlabel(f'U{plot_vecs[2]}', labelpad=-6)
    sax = ax.scatter(*geom.u[:, plot_vecs].t().cpu(), c=gip[:, 1], cmap='hsv', s=3)
    ax.set_title(f'{plot_vecs}: Az')

    ax = fig.add_subplot(len(plot_vec_list), 2, 2 + ii*2, projection='3d')
    ax.view_init(view_dl, view_az), ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
    ax.set_xlabel(f'U{plot_vecs[0]}', labelpad=-6), ax.set_ylabel(f'U{plot_vecs[1]}', labelpad=-6), ax.set_zlabel(f'U{plot_vecs[2]}', labelpad=-6)
    sax = ax.scatter(*geom.u[:, plot_vecs].t().cpu(), c=gip[:, 0], cmap='hsv', s=3)
    ax.set_title(f'{plot_vecs}: DLA')
plt.show(), plt.close(fig)


# plot in CIDM coords
view_az, view_dl = 20, 40
tan_ss = slice(None, None, 7)
for view_az in range(0, 180, 1000):
    fig = plt.figure(figsize=(10, 18))
    plot_vecs = [1, 2, 3]
    plot_vecs = [3, 4, 1]
    ax = fig.add_subplot(4, 2, 1, projection='3d')
    sax = ax.scatter(*geom.u[:, plot_vecs].t().cpu(), c=gip[:, 1], cmap='hsv', s=3)
    scbar = fig.colorbar(sax, ticks=list(range(int(gip[:, 1].min()), int(gip[:, 1].max() + 1), 5)),
                         shrink=0.7, fraction=0.03, panchor=False)
    scbar.ax.set_xlabel('Az', labelpad=10)
    ax.view_init(view_dl, view_az), ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
    ax.set_xlabel(f'U{plot_vecs[0]}', labelpad=-6), ax.set_ylabel(f'U{plot_vecs[1]}', labelpad=-6), ax.set_zlabel(f'U{plot_vecs[2]}', labelpad=-6)
    ax.set_title('Az')

    ax = fig.add_subplot(4, 2, 2, projection='3d')
    sax = ax.scatter(*geom.u[:, plot_vecs].t().cpu(), c=gip[:, 0], cmap='hsv', s=3)
    scbar = fig.colorbar(sax, ticks=list(range(int(gip[:, 0].min()), int(gip[:, 0].max() + 1), 5)),
                         shrink=0.7, fraction=0.03, panchor=False)
    scbar.ax.set_xlabel('DL', labelpad=10)
    ax.view_init(view_dl, view_az), ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
    # ax.set_xlabel(f'U{plot_vecs[0]}', labelpad=-6), ax.set_ylabel(f'U{plot_vecs[1]}', labelpad=-6), ax.set_zlabel(f'U{plot_vecs[2]}', labelpad=-6)
    ax.set_title('DLA')

    num_vecs = 1  # len of uvec, assume 1 for now
    from PySEC.sec_utils import reshape_fortran

    vec_fields = []
    for uvec in range(6):
        umatrix = reshape_fortran(geom.h1.t() @ geom.u1[:, uvec], (num_vecs, geom.n1, geom.n1))
        vectorfield = torch.tensordot(geom.u[:, :geom.n1], umatrix, dims=[[-1], [-2]])
        vec_fields.append(vectorfield)

        ax = fig.add_subplot(4, 2, 3 + uvec, projection='3d')
        sax = ax.scatter(*geom.u[tan_ss, plot_vecs].t().cpu(), c='gray', alpha=0.6, s=5)
        ax.scatter(*geom.u[jidx, plot_vecs].t().cpu(), 'blue', s=40)
        ax.view_init(view_dl, view_az), ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
        tan_len_fac = 1.e0 / torch.linalg.norm(vectorfield).item()
        tax = ax.quiver(*geom.u[tan_ss, plot_vecs].t().cpu(), *vectorfield[tan_ss, 0, plot_vecs].t().cpu(),
                        length=2 * tan_len_fac, normalize=False, color='red', arrow_length_ratio=0.4, alpha=0.6, )
        ax.set_title(f'|Del1 Tan({uvec})|: {torch.linalg.norm(vectorfield):.2e}')
        ax.set_xlabel(f'U{plot_vecs[0]}', labelpad=-6), ax.set_ylabel(f'U{plot_vecs[1]}', labelpad=-6), ax.set_zlabel(f'U{plot_vecs[2]}', labelpad=-6)

    fig.tight_layout()
    plt.show(), plt.close(fig)


uvec = 7
usize = 20
umatrix = reshape_fortran(geom.h1.t() @ geom.u1[:, uvec], (num_vecs, geom.n1, geom.n1))
umatrices = torch.permute(geom.h1.t() @ geom.u1[:, :usize], [1, 0])
umatrices = umatrices.reshape((usize, geom.n1, geom.n1)).transpose(1, 2)
torch.allclose(umatrices[uvec].ravel(), umatrix.ravel())
# vectorfields = torch.tensordot(
#     geom.u[:, :geom.n1],
#     torch.tensordot(umatrices, geom.Xhat[:geom.n1], dims=1),
#     dims=[[-1], [-2]])
vectorfields = torch.tensordot(
    geom.u[:, :geom.n1],
    umatrices @ geom.Xhat[:geom.n1],
    dims=[[-1], [-2]])



# plot in intrinsic coords
view_az, view_dl = 20, 40
tan_ss = slice(None, None, 7)
for view_az in range(0, 180, 1000):
    fig = plt.figure(figsize=(10, 18))
    for uvec in range(12):
        vectorfield = vectorfields[:, uvec]
        ax = fig.add_subplot(4, 3, 1 + uvec, projection='3d')
        ax.scatter(*data[:, tan_ss], c='gray', s=5, alpha=0.6)
        ax.view_init(view_dl, view_az), ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
        tan_len_fac = 1.e1 / torch.linalg.norm(vectorfield).item()
        tax = ax.quiver(*data[:, tan_ss], *vectorfield[tan_ss, :].t(),
                        length=2 * tan_len_fac, normalize=False, color='red', arrow_length_ratio=0.4, alpha=0.6, )
        ax.set_title(f'|Del1 Tan({uvec})|: {torch.linalg.norm(vectorfield):.2e}')

    fig.canvas.draw(), fig.tight_layout() # tight layout error hack
    plt.show(), plt.close(fig)

    # vfs = torch.stack(vec_fields).squeeze()
    # vfs = torch.permute(vfs, [1, 0, 2])
    # ub, db, vhb = torch.linalg.svd(vfs, full_matrices=False)

#
# vfs = torch.stack(vec_fields).squeeze()
# vfs = torch.permute(vfs, [1, 0, 2])
# ub, db, vhb = torch.linalg.svd(vfs, full_matrices=False)
# # plot svd of vectorfields in CIDM coords
# view_az, view_dl = 20, 40
# tan_ss = slice(None, None, 2)
# for view_az in range(0, 180, 1000):
#     fig = plt.figure(figsize=(10, 18))
#     plot_vecs = [1, 2, 5]
#     ax = fig.add_subplot(4, 2, 1, projection='3d')
#     sax = ax.scatter(*geom.u[:, plot_vecs].t().cpu(), c=gip[:, 1], cmap='hsv', s=3)
#     scbar = fig.colorbar(sax, ticks=list(range(int(gip[:, 1].min()), int(gip[:, 1].max() + 1), 5)),
#                          shrink=0.7, fraction=0.03, panchor=False)
#     scbar.ax.set_xlabel('Az', labelpad=10)
#     ax.view_init(view_dl, view_az), ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
#     ax.set_title('Az')
#
#     ax = fig.add_subplot(4, 2, 2, projection='3d')
#     sax = ax.scatter(*geom.u[:, plot_vecs].t().cpu(), c=gip[:, 0], cmap='hsv', s=3)
#     scbar = fig.colorbar(sax, ticks=list(range(int(gip[:, 0].min()), int(gip[:, 0].max() + 1), 5)),
#                          shrink=0.7, fraction=0.03, panchor=False)
#     scbar.ax.set_xlabel('DL', labelpad=10)
#     ax.view_init(view_dl, view_az), ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
#     ax.set_title('DLA')
#
#     num_vecs = 1  # len of uvec, assume 1 for now
#     from PySEC.sec_utils import reshape_fortran
#
#     for uvec in range(6):
#         vectorfield = vhb[:, uvec].unsqueeze(1)
#         ax = fig.add_subplot(4, 2, 3 + uvec, projection='3d')
#         sax = ax.scatter(*geom.u[tan_ss, plot_vecs].t().cpu(), c='gray', alpha=0.6, s=5)
#         ax.scatter(*geom.u[jidx, plot_vecs].t().cpu(), 'blue', s=40)
#         ax.view_init(view_dl, view_az), ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
#         tan_len_fac = 1.e0 #/ torch.linalg.norm(vectorfield).item()
#         # tan_len_fac = torch.linalg.norm(vectorfield).item()
#         tax = ax.quiver(*geom.u[tan_ss, plot_vecs].t().cpu(), *vectorfield[tan_ss, 0, plot_vecs].t().cpu(),
#                         length=2 * tan_len_fac, normalize=False, color='red', arrow_length_ratio=0.4, alpha=0.6, )
#         ax.set_title(f'Del1 Tan({uvec})')
#
#     fig.tight_layout()
#     plt.show(), plt.close(fig)


debug_var = 1