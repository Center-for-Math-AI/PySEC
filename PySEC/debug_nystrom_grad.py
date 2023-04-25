import os
import torch
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from PySEC import generate_data as gd
from PySEC.nystrom_cidm import cidm, nystrom, nystrom_grad
from PySEC.del1 import del1_as_einsum
from PySEC.sec_utils import reshape_fortran


import matlab
import matlab.engine as me
eng = me.start_matlab()
sec_path = os.path.join(os.path.split(os.path.dirname(__file__))[0], "matlab")
eng.addpath(sec_path)


def dataclass2matlab_dict(dclass):  # dclass = dataclass

    # lots of shennanigans to convert to a matlab struct with correct data ordering
    not_transpose = ["u"]  # list of vars to not transpose (move to fortran order)
    ret = {mem: vars(dclass)[mem] for mem in vars(dclass) if not mem.startswith("__")}
    for key, val in ret.items():
        new_val = val
        if isinstance(val, torch.Tensor):
            if new_val.is_sparse:  # all sparses are in fortran order
                new_val = val.coalesce().to_dense()

            if len(new_val.shape) == 1:  # new axis
                new_val = new_val.reshape((1, new_val.shape[0]))

            if key not in not_transpose:
                new_val = new_val.T

            new_val = matlab.double(new_val.clone().numpy())

        elif isinstance(new_val, float):
            new_val = matlab.double(new_val)

        ret[key] = new_val

    return ret



# test = gd.generate_circle(100)
test = gd.generate_noisy_circle(21, 0.01)
points = test[0].t()
t = test[1]
angs = torch.atan2(t[1], t[0])
xs = slice(None, None, 11)
xsub = points[xs, :]
tsub = angs[xs]


# def cidm(x, nvars=None, k=None, k2=None, tuning_method=None, **knn_args):
u0, l0, peq0, qest0, eps0, dim0, KP = cidm(points, k=7, k2=5)
Xhat = u0.T @ torch.diag(peq0) @ \
            points.view(points.shape[0], -1).to(dtype=u0.dtype, device=u0.device)

cidm_ss = slice(1, 4)
xhat2 = u0.T @ torch.diag(peq0) @ u0[:, cidm_ss]
xhat3 = u0.T @ torch.diag(peq0) @ u0[:, :]

n1 = min(80, KP.k)  # use kNN size capped at 80
# n1 = l.shape[0] - 10
u1, l1, d1, _, h1, _ = del1_as_einsum(u0, l0, torch.diag(peq0), n1)


u2, peq2, qest2, gradu, debug = nystrom_grad(xsub, KP)
debug_var = 1

retm = eng.NystromCIDMgrad(
    matlab.double(xsub.clone().numpy().T),
    eng.struct(dataclass2matlab_dict(KP)),
    nargout=5,
)
# u2m, peq2m, qest2m, gradum, debugm = retm
u2m = torch.as_tensor(retm[0], dtype=u2.dtype)
gradum = torch.as_tensor(retm[3], dtype=gradu.dtype).permute([1, 2, 0])
perm_list = list(range(1,len(debug.shape))) + [0,]
# debugm = torch.as_tensor(retm[4], dtype=float) #dtype=debug.dtype).permute(perm_list)
# debugm = torch.as_tensor(retm[4], dtype=debug.dtype).permute(perm_list)
# debugm = reshape_fortran(torch.as_tensor(retm[4], dtype=debug.dtype), debug.shape)
debugm = torch.as_tensor(retm[4], dtype=debug.dtype).reshape(debug.shape)


# A = torch.linalg.solve(u0[:, 1:3], points[:, 0:2])

# fig, ax = plt.subplots()
# cols = ['tab:green', 'tab:red']
# for ii in range(A.shape[1]):
#     ax.scatter(angs, points[:, ii], label=f"data {ii}", c=cols[ii])
#     # ax.plot(angs, u0[:, ii] * A[:, ii], label=f"eig {ii}")
# ax.legend()
# plt.show(), plt.close(fig)


# fig, ax = plt.subplots()
# for ii in range(xsub.shape[1]):
#     ax.plot(tsub, xsub[:, ii])
#     ax.plot(tsub, u2[:, ii+1] * A)
debug_var = 1

fig, aax = plt.subplots(3, 3, figsize=(12,12))
for ii, ax in enumerate(aax.flat):
    ax.scatter(xsub[:,0], xsub[:,1])
    gradui = gradu[:, ii + 1]
    ax.quiver(xsub[:, 0], xsub[:, 1], gradui[:, 0], gradui[:, 1], color='tab:red')
fig.suptitle('Python')
plt.show(), plt.close(fig)


fig, aax = plt.subplots(3, 3, figsize=(12,12))
for ii, ax in enumerate(aax.flat):
    ax.scatter(xsub[:,0], xsub[:,1])
    gradui = gradum[:, ii + 1]
    ax.quiver(xsub[:, 0], xsub[:, 1], gradui[:, 0], gradui[:, 1], color='tab:red')
fig.suptitle('Matlab')
plt.show(), plt.close(fig)

torch.allclose(gradu, gradum)
torch.linalg.norm(gradu - gradum)
torch.linalg.norm(gradu)
torch.linalg.norm(gradum)


cof = 0.01
cof = 1
iof = 0
# tmp = KP.X[dxi[:, 1:KP.k2]]
# debug = reshape_fortran(tmp, tmp.shape)
fig, aax = plt.subplots(2, 2, figsize=(8, 8))
for ii, ax in enumerate(aax.flat):
    ax.scatter(debug[:, ii+iof, 0], debug[:, ii+iof, 1], label='python', alpha=0.4)
    # ax.scatter(cof*tmp[:, ii+iof, 0], cof*tmp[:, ii+iof, 1], label='test', alpha=0.4)
    ax.plot(debugm[:, ii+iof, 0], debugm[:, ii+iof, 1], color='tab:red', linestyle='', marker='o', markerfacecolor='none', label='matlab', alpha=0.6)
    # ax.plot(debugm[0, :, ii+iof], debugm[1, :, ii+iof], color='tab:red', linestyle='', marker='o', markerfacecolor='none', label='matlab', alpha=0.6)
    ax.legend()
plt.show(), plt.close(fig)

torch.allclose(debug.ravel(), debugm.ravel(), atol=1e-14, rtol=1e-14)

spv, spi = tmp.ravel().sort()
smv, smi = debugm.ravel().sort()


fig, aax = plt.subplots(1, 2, figsize=(10,8))
aax[0].matshow(debug, label='python')
aax[0].set_title('python')
aax[1].matshow(debugm-1, label='matlab')
aax[1].set_title('matlab')
plt.show(), plt.close(fig)

torch.allclose(debug.to(float), debugm.to(float)-1.)


iof = 0
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple']
fig, aax = plt.subplots(1, 2, figsize=(10, 8))
for ii, ax in enumerate(aax.flat):
    ax.plot(xsub[ii, 0], xsub[ii, 1], color=colors[0], linestyle='', marker='o', label='data', alpha=0.8)
    ax.plot(debug[ii, :, 0], debug[ii, :, 1], color=colors[0], linestyle='', marker='v', markerfacecolor='none', label='python', alpha=0.6)
    ax.plot(debugm[ii, :, 0], debugm[ii, :, 1], color=colors[1], linestyle='', marker='^', markerfacecolor='none', label='matlab', alpha=0.6)
    ax.legend()
plt.show()


debug_var = 1
