import torch
from torch import permute as tp
from torch.sparse import mm as smsm
from math import ceil, log
from PySEC.distance_funcs import self_knn_expensive, knn_expensive
from PySEC.sec_utils import repmat, estimate_dimension, sparse_diag
from dataclasses import dataclass

# maybe we get non-exact mapping from linear dependence in the dataset?
# in which case we could try to remove linear dependence and create a linearly
# independent basis using the QR decomp method:
# https://math.stackexchange.com/questions/3340443/remove-linearly-dependent-vectors-from-a-matrix


@dataclass
class DiffusionKernelData:
    epsilon: float = 0
    dim: int = 0
    k: int = 0
    k2: int = 0
    rho: torch.Tensor = None
    d: torch.Tensor = None
    X: torch.Tensor = None
    peq: torch.Tensor = None
    u: torch.Tensor = None
    l: torch.Tensor = None
    lheat: torch.Tensor = None

    def to(self, dest):
        """ send tensors to type or device """
        self.rho = self.rho.to(dest)
        self.d = self.d.to(dest)
        self.X = self.X.to(dest)
        self.peq = self.peq.to(dest)
        self.u = self.u.to(dest)
        self.l = self.l.to(dest)
        self.lheat = self.lheat.to(dest)
        return self

# end class DiffusionKernelData:


def cidm(x, nvars=None, k=None, k2=None, tuning_method=None, **knn_args):

    # @TODO: provide ability to pass in knn data, so a user can do arbitrary metrics (for intrinsic)

    N, n = x.shape[0], x.shape[1]
    if k2 is None:
        k2 = ceil(log(N))
    if k is None:
        k = ceil(log(N) ** 2)
    if nvars is None:
        nvars = 2 * k

    KP = DiffusionKernelData()
    KP.X = x
    KP.k = k
    KP.k2 = k2

    dx, dxi = self_knn_expensive(x, k, **knn_args)

    rho = torch.mean(dx[:, 1:k2], dim=1)  #
    KP.rho = rho
    dx = dx * dx / (repmat(rho, [k]) * rho[dxi])

    if tuning_method is None:
        epsilon = torch.mean(dx[:, 1:k2]).item()
    else:
        raise NotImplementedError()
    KP.epsilon = epsilon

    # dim, ddim = estimate_dimension(dx, epsilon)
    dim, ddim = estimate_dimension(dx, epsilon)
    KP.dim = dim.item()

    dx = torch.exp(-dx / (2 * epsilon))

    coo = torch.stack((repmat(torch.arange(N, device=x.device), [k]).reshape(-1), dxi.reshape(-1)))
    d_sparse = torch.sparse_coo_tensor(coo, dx.reshape(-1), [N, N])
    KP.d = d_sparse.clone()
    d_sparse = 0.5 * (d_sparse + d_sparse.t())
    d_sparse = d_sparse.coalesce()

    # CIDM normalization
    peq = torch.sparse.sum(d_sparse, dim=1).to_dense()  #
    KP.peq = peq
    # peq = torch.sum(dx, dim=1)
    Dinv = sparse_diag(torch.pow(peq, -0.5))

    tmp = smsm(Dinv, d_sparse)
    d_sparse = smsm(tmp, Dinv)

    d_sparse = 0.5 * (d_sparse + d_sparse.t())

    # @TODO: do without .to_dense(), may need to use scipy.sparse.linalg.eigsh
    # or else write my own using a QR solver
    if 3 * nvars > N:  # can't use torch.lobpcg, cutoff around N=156
        # if True:
        sigma = 1.01
        l, u = torch.linalg.eigh(
            d_sparse.to_dense() - sigma * torch.eye(d_sparse.shape[0], device=d_sparse.device)
        )
        l, sidx = torch.topk(l + sigma, nvars, largest=True)  # mimic matlab eigs()
        u = u[:, sidx]  # sort and subset eigenvecs
        KP.lheat = l  # heat kernel eigs

        las, asidx = torch.sort(torch.abs(KP.lheat), descending=True)  # absolute sort
        KP.l = torch.abs(torch.log(KP.lheat[asidx])) / epsilon
        KP.u = Dinv @ u[:, asidx]  # absolute sort
    else:
        # could modify d_sparse in place for lobpcg then reverse operation after...
        # will lose precision for distances << 1
        # if the eigs are ordered as 1 = λ0 > |λ1| >= |λ2| >= ..., then only returning
        # the largest can be incorrect
        sigma = 1.01
        d_eig = d_sparse - sparse_diag(torch.full([N], sigma,
                                                  dtype=d_sparse.dtype,
                                                  device=d_sparse.device))
        l, u = torch.lobpcg(d_eig, k=nvars, largest=True)
        l = l.clone() + torch.tensor(sigma, dtype=l.dtype, device=l.device) # lobpcg returns a view, can't add in place
        KP.lheat = l
        KP.l = torch.abs(torch.log(l)) / epsilon
        KP.u = Dinv @ u

    qest = peq / (torch.pow(rho, dim) * N * (2 * torch.pi * epsilon) ** (0.5 * dim))

    return KP.u, KP.l, peq, qest, epsilon, dim, KP
# end def cidm


def nystrom(x, KP, **knn_args):
    # NB: we assume x must be batched!

    #@TODO: this will be a bottleneck eventually and could be much cheaper
    # if you saved the ball tree from knn, in which case you just follow
    # the path to the nearest neighbor by sampling your distance to a
    # random point in the tree, should be logN instead of N

    if len(x.shape) < len(KP.X.shape):
        shape_diff = len(KP.X.shape) - len(x.shape)
        xknn = x.view(*[1,]*shape_diff, *KP.X.shape[shape_diff:])
    else:
        xknn = x

    N, n = xknn.shape[0], xknn[0].nelement()
    k = KP.k
    k2 = KP.k2

    dx, dxi = knn_expensive(xknn, KP.X, k, **knn_args)

    # CkNN Normalization
    rho = torch.mean(dx[:, 1:k2], dim=1)
    dx = dx * dx / (repmat(rho, [k]) * KP.rho[dxi])

    # RBF kernel
    dx = torch.exp(-dx / (2 * KP.epsilon))
    coo = torch.stack((repmat(torch.arange(N, device=x.device), [k]).reshape(-1),
                       dxi.reshape(-1)))
    d_sparse = torch.sparse_coo_tensor(coo, dx.reshape(-1), [N, KP.X.shape[0]])
    d_sparse = d_sparse.coalesce()

    # peq = torch.sparse.sum(d_sparse, dim=1).to_dense()  #
    peq = torch.sum(dx, dim=1)  # don't need sparse
    qest = peq / (
        N * torch.pow(rho, KP.dim) * (2 * torch.pi * KP.epsilon) ** (0.5 * KP.dim)
    )

    D = sparse_diag(1 / peq)
    tmp = smsm(D, d_sparse)
    d_sparse = tmp
    d_sparse = d_sparse.coalesce()

    # Linv = sparse_diag(1 / KP.lheat)
    #u = smsm(Linv.t(), smsm(d_sparse, KP.u).t()).t()
    u = torch.einsum('ij,j->ij', d_sparse @ KP.u, 1./KP.lheat)

    return u, peq, qest
# end def nystrom

def nystrom_grad(x, KP, **knn_args):

    if len(x.shape) < len(KP.X.shape):
        shape_diff = len(KP.X.shape) - len(x.shape)
        xknn = x.view(*[1,]*shape_diff, *KP.X.shape[shape_diff:])
    else:
        xknn = x

    N, n = xknn.shape[0], xknn[0].nelement()
    k = KP.k
    k2 = KP.k2

    dx, dxi = knn_expensive(xknn, KP.X, k, **knn_args)

    # CkNN Normalization
    rho = torch.mean(dx[:, 1:k2], dim=1)
    dx = dx * dx / (repmat(rho, [k]) * KP.rho[dxi])

    # M, n = KP.X.shape
    # N, n = x.shape
    # N, k = dxi.shape
    # N, k, n = KP.X[dxi].shape
    amb_shape = xknn.shape[1:]
    eps = torch.finfo(x.dtype).eps
    # v1 = xknn.view((N, 1, n)).expand(N, k, n) - KP.X[dxi].view(N, k, -1)
    v1 = xknn[:, None, :] - KP.X[dxi] #.view([N, k, n]) # shape [N, k, n], use view to flatten ambient just in case
    v1 = v1 / (torch.sum(v1*v1, dim=-1) + eps)[:, :, None] # shape [N, k, n]

    # N, k2-1, n = KP.X[dxi[:, 1:k2]].shape
    v2 = xknn[:, None, :] - KP.X[dxi[:, 1:k2]] # shape [N, k2-1, n]
    # v2 = xknn.view([N, 1, n]).expand([N, k2-1, n]) - KP.X[dxi[:, 1:k2]].view([N, k2-1, n]) # shape [N, k2-1, n]
    # debug = KP.X[dxi[:, 1:k2]].clone() #.view([N, k2-1, n]).clone()
    # debug = dxi[:, 1:k2].clone()
    v2 = torch.sum(v2, dim=1).view([N, 1, n]).expand([N, k, n]) # shape [N, k, n]
    v2 = v2 / (torch.sum(v2*v2, dim=-1) + eps)[:, :, None] # shape [N, k, n]
    # v2 = v2.view([N, 1, n]).expand([N, k, n])

    # gradd = dx[:, :, None] * (2*v1 - v2[:, None, :]) # shape [N, k, n]
    gradd = dx[:, :, None] * (2*v1 - v2) # shape [N, k, n]


    # size(X) = [n,M]
    # size(KP.X) = [n,N]
    # size(KP.X(:, inds)) = [n, k * M]
    # v1 =
    #   repmat(X, [1 1 k]) -
    #   permute(
    #       reshape(KP.X(:, inds), [n k N]),
    #   [1 3 2]);
    # shape1: [n M k] - shape2: [n N k]
    # v1 = v1. / repmat(sum(v1. ^ 2) + eps, [n 1 1]);


    # v2 = repmat(X, [1 1 k2 - 1]) - permute(reshape(KP.X(:, inds(2: k2,:))',[n k2-1 N]),[1 3 2]);
    # shape1: n, N, k2-1 - shape2: n, N, k2-1
    # v2 = repmat(sum(v2, 3), [1 1 k]);
    # v2 = v2. / repmat(sum(v2. ^ 2) + eps, [n 1 1]);
    # gradd = repmat(reshape(d',[1 N k]),[n 1 1]).*(2*v1 - v2); % shape [n, N, k]

    # RBF kernel
    dx = torch.exp(-dx / (2 * KP.epsilon)) # shape [N, k]

    m = KP.lheat.shape[0] # eig size
    dx_sumk = dx.sum(dim=1)

    # gradd = dx[:, :, None] * (2*v1 - v2) # shape [N, k, n]
    gradu = torch.sum(dx[:, :, None] * gradd, dim=1) # sum out k, shape [N, n]
    gradu = gradu / torch.sum(dx, dim=1)[:, None] # shape [N, n]

    gradu = gradu[:, None, :] - gradd # shape [N, k, n]

    uu = repmat(KP.u[dxi, :], [n,]) # shape [N, k, m, n]

    gradu = torch.sum(uu * (dx[:, :, None] * gradu)[:, :, None, :], dim=1)
    gradu = gradu / torch.sum(dx, dim=1)[:, None, None]

    gradu = gradu / KP.lheat[None, :, None]


    coo = torch.stack((repmat(torch.arange(N, device=x.device), [k]).reshape(-1),
                       dxi.reshape(-1)))
    d_sparse = torch.sparse_coo_tensor(coo, dx.reshape(-1), [N, KP.X.shape[0]])
    d_sparse = d_sparse.coalesce()

    # peq = torch.sparse.sum(d_sparse, dim=1).to_dense()  #
    peq = torch.sum(dx, dim=1)  # don't need sparse
    qest = peq / (
        N * torch.pow(rho, KP.dim) * (2 * torch.pi * KP.epsilon) ** (0.5 * KP.dim)
    )

    D = sparse_diag(1 / peq)
    tmp = smsm(D, d_sparse)
    d_sparse = tmp
    d_sparse = d_sparse.coalesce()

    # Linv = sparse_diag(1 / KP.lheat)
    #u = smsm(Linv.t(), smsm(d_sparse, KP.u).t()).t()
    u = torch.einsum('ij,j->ij', d_sparse @ KP.u, 1./KP.lheat)

    # debug = tmpret
    return u, peq, qest, gradu
# end def nystrom
