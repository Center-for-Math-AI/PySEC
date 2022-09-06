import torch
from scipy.linalg import eigh
from math import ceil, log, pi
from PySEC.distance_funcs import pdist2
#from scipy.spatial.distance import cdist
#from PySEC.sec_utils import eig_wrap


def del0(x, n, epsilon=None):
    """
    Construction of the 0-Laplacian using the Diffusion Maps algorithm
    :param x: data, shape == [ambient, num_examples]
    :param n: how many eigenvectors to use
    :param epsilon: optional bandwidth
    :return: u, l, D
    """
    # xtn = x.T.numpy()
    # dmat = torch.as_tensor(cdist(xtn, xtn)) # scipy cdist is very close to matlab pdist2
    # dmat = torch.cdist(x.T, x.T) # only does p-norm distance, default is 2.0
    # is torch.cdist non-deterministic???
    #dmat = self_pair_dist_p2(x)
    dmat = pdist2(x)

    if epsilon is None:
        k = 1 + ceil(log(n))
        epsilon = torch.topk(dmat, k, largest=False)[0]
        epsilon = epsilon[:, -1].mean()

    eps2 = epsilon * epsilon

    dmat = torch.exp(-dmat * dmat / (4.0 * eps2))
    dmat = 0.5 * (dmat + dmat.T)

    dtmp = 1.0 / dmat.sum(dim=1)  # get a vector instead of diag
    dmat = torch.einsum(
        "ab,b->ab", dmat, dtmp
    )  # multiply a vector as if it's a diag matrix
    dmat = torch.einsum(
        "a,ab->ab", dtmp, dmat
    )  # multiply a vector as if it's a diag matrix
    # dtmp = torch.diag(1. / dmat.sum(dim=1)) # take a vector and give a diag matrix
    # dmat = torch.matmul(dmat, dtmp)
    # dmat = torch.matmul(dtmp, dmat)
    dmat = 0.5 * (dmat + dmat.T)

    dsum = torch.diag(dmat.sum(dim=1))

    # l, u = eig_wrap(dmat, dsum, n)
    ln, un = eigh(
        dmat.cpu().numpy(),
        dsum.cpu().numpy(),
        eigvals_only=False,
        subset_by_index=[dmat.shape[0] - n, dmat.shape[0] - 1],
    )
    l, u = torch.as_tensor(ln[::-1].copy()), torch.as_tensor(
        un[:, ::-1].copy()
    )  # eigh returns ascending order
    llog = -torch.log(l) / eps2
    return u.to(x.dtype).to(x.device), \
           llog.to(x.dtype).to(x.device), \
           dsum.to(x.dtype).to(x.device)
# end def del0
