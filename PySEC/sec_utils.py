import torch
import scipy.linalg
from scipy.linalg import eigh, eig
import numpy as np


def reshape_fortran(x: torch.Tensor, shape: list):
    """
    reshape in fortran order since torch reshape doesn't support
    """
    if 0 < len(x.shape):
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def repmat(x: torch.Tensor, new_dims: list):
    ret = x.view((*list(x.shape), *torch.ones(len(new_dims), dtype=int).tolist()))
    return ret.expand((*list(x.shape), *new_dims))


def sparse_diag(vec, mat_shape=None, diag=0):

    if mat_shape is None:
        mat_shape = [vec.shape[0], vec.shape[0]]
    min_shape = min(mat_shape)  # handles rectangular
    diag_ids = torch.arange(min_shape, device=vec.device)

    if diag == 0:
        coo = torch.stack((diag_ids, diag_ids))
        return torch.sparse_coo_tensor(coo, vec, mat_shape)
    else:
        raise NotImplementedError  # handle off center diag


def outer_sum(x: torch.Tensor, y: torch.Tensor):
    ret = x.ravel().view(-1, 1) + y.ravel()
    return ret.view(*x.shape, *y.shape)


def eig_wrap(a, b, n=None, sort_type="largest mag"):
    """
    wrapper for generalized eigen solver, torch preferred if possible
    a x = l b x, a and b are matrices, x is vec, l is diag eigen value matrix
    :param a: primary
    :param b: companion matrix
    :param n: how many eigen pairs to return
    :return: eigenvalues, eigenvectors as torch tensors
    """
    k = a.shape[0] if n is None else n
    descending = True if sort_type.lower().startswith("l") else False

    w, vr = eig(a.cpu().numpy(), b=b.cpu().numpy(), left=False, right=True)
    wt = torch.as_tensor(w.real, dtype=a.dtype)
    if "mag" in sort_type.lower() or "abs" in sort_type.lower():
        wt = torch.abs(wt)
    L, isort = torch.sort(wt, descending=descending)
    L = L[:k]
    U = torch.as_tensor(vr.real, dtype=a.dtype)[:, isort[:k]]

    return L.to(a.device), U.to(a.device)
# end def eig_wrap


def get_degen_patt(vals, atol=1.0e-8, rtol=1.0e-6):
    averages = []
    std_devs = []
    degen_patts = []

    if len(vals) == 1:
        averages.append(vals[0])
        std_devs.append(0.0)
        degen_patts.append(1)

    elif len(vals) > 1:

        dlist = [vals[0]]
        for ii, val in enumerate(vals[1:]):
            if abs(dlist[0] - val) < rtol * max(abs(dlist[0]), abs(val)) + atol:
                dlist.append(val)
            else:
                ndlist = torch.as_tensor(dlist)
                degen_patts.append(len(dlist))
                averages.append(ndlist.mean())
                std_devs.append(ndlist.std(unbiased=False))
                dlist = [val]

        # now do the last list
        ndlist = torch.as_tensor(dlist)
        degen_patts.append(len(dlist))
        averages.append(ndlist.mean())
        std_devs.append(ndlist.std(unbiased=False))

    return degen_patts, averages, std_devs
# end def get_degen_patt(vals, degen_tol):


def compare_eigenpairs(val1, vec1, val2, vec2, atol=1.0e-8, rtol=1.0e-6,
                       verbose=False, skip_last=1):
    """
    compare two eigen decomps and handle degenerate subspaces by checking if
    one subspace can be fully represented by the other subspace
    :param val1: vector of eigenvalues
    :param vec1: vector of eigenvectors (matrix), assume orthonormal
    :param val2: vector of eigenvalues
    :param vec2: vector of eigenvectors (matrix), assume orthonormal
    :param atol: absolute tolerance for torch.allclose
    :param rtol: relative tolerance for torch.allclose
    :param verbose: whether to fail silently or with message
    :param skip_last: how many of the vectors to skip comparing at the end
    :return: boolean
    """

    if val1.shape != val2.shape or vec1.shape != vec2.shape:
        if verbose:
            print("\n **compare_eigenpairs failed at eigenvalue/vector shape check")
        return False

    vals_same = torch.allclose(val1, val2, atol=atol, rtol=rtol)
    if not vals_same:
        if verbose:
            print("\n **compare_eigenpairs failed at eigenvalue value comparison check")
        return False

    # now assume the same degeneracy pattern of the two decomps and find val1
    if isinstance(skip_last, int) and skip_last > 0:
        degen_patt, degen_avg, degen_std = get_degen_patt(val1[:-skip_last],
                                                          atol=atol, rtol=rtol)
    else:
        degen_patt, degen_avg, degen_std = get_degen_patt(val1,
                                                          atol=atol, rtol=rtol)
    iv = 0
    for ip, patt in enumerate(degen_patt):
        # due to degeneracy, and eigenvector phase indeterminacy compare with QR
        ss = slice(iv, iv + patt)
        iv += patt
        norms1 = torch.linalg.norm(vec1[ss], dim=-1)
        norms2 = torch.linalg.norm(vec2[ss], dim=-1)
        if norms1.sum() < atol and norms2.sum() < atol:
            continue  # skip zero vectors
        normm = torch.einsum("i,j->ij", norms1, norms2)
        overs = torch.einsum("ik,jk->ij", vec1[ss], vec2[ss]) / normm
        p12_rank = torch.sum(overs * overs).item()
        vec_same = abs(patt - p12_rank) < rtol * patt + atol
        if not vec_same:
            if verbose:
                print(
                    "\n **compare_eigenpairs failed at eigenvector value comparison check"
                )
                print(
                    f"    degen_patt: {patt}, degen_val: {degen_avg[ip]}, overlap_rank: {p12_rank}"
                )
            return False

    return True
# end def compare_eigenpairs


def estimate_dimension(d, epsilon, diagonals=None):

    device = d.device
    if diagonals is None:
        diagonals = 1

    N = d.shape[1]
    dx = 1.0e-4
    opdx = 1.0 + dx
    omdx = 1.0 - dx
    if d.is_sparse:
        dsum = d.values()
    else:
        dsum = d
    ds1 = torch.exp(-dsum[dsum > 1.0e-6] / (2 * epsilon * opdx)).sum() + N * diagonals
    ds2 = torch.exp(-dsum[dsum > 1.0e-6] / (2 * epsilon * omdx)).sum() + N * diagonals

    dim = 2 * torch.log(ds1 / ds2) / torch.log(torch.tensor(opdx / omdx, device=d.device, dtype=torch.float64))

    ds = torch.exp(-dsum[dsum > 0] / (2 * epsilon)).sum() + N * diagonals
    ddim = 2 * torch.log(ds1 * ds2 / (ds * ds)) / (dx * dx) + dim

    return dim, ddim
# end def estimate_dimension
