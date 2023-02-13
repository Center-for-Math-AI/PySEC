import torch
import numpy as np
from PySEC.distance_funcs import *
from PySEC.sec_utils import reshape_fortran, get_degen_patt
from PySEC.generate_data import laplacian_eig_truth
import scipy.spatial.distance as sdist
from sklearn.neighbors import NearestNeighbors

atol = 1.0e-14
rtol = 1.0e-10


def test_self_pair_dist():
    x = torch.arange(0, 10, 2).reshape(-1, 1) * 1.0
    x *= x  # square to make it non-linear spacing so result has less symmetry
    dx = self_pair_dist_p2(x.T)

    assert dx.shape[0] == dx.shape[1] == x.shape[0]
    assert torch.allclose(x[:, 0], dx[0], atol=atol, rtol=rtol)

    dxs = sdist.cdist(x, x)
    assert torch.allclose(dx.to(float), torch.tensor(dxs))


def test_pdist2_self():
    x = torch.arange(0, 10, 2).reshape(-1, 1) * 1.0
    x *= x  # square to make it non-linear spacing so result has less symmetry
    dx = pdist2(x)

    assert dx.shape[0] == dx.shape[1] == x.shape[0]
    assert torch.allclose(x[:, 0], dx[0], atol=atol, rtol=rtol)

    dxs = sdist.cdist(x, x)
    assert torch.allclose(dx.to(float), torch.tensor(dxs, dtype=float), atol=atol, rtol=rtol)

    # test self distance optional arg is consistent with two args
    dxx = pdist2(x, x)
    assert torch.allclose(dxx.to(float), dx.to(float), atol=atol, rtol=rtol)

def test_pdist2_xy():
    x = torch.arange(0, 10, 2).reshape(-1, 1) * 1.0
    x = x.to(float)
    x *= x  # square to make it non-linear spacing so result has less symmetry
    y = torch.arange(0, 8).reshape(-1, 1) * 1.0 + 2.0
    y = y.to(float)
    y = torch.sqrt(y)  # make it non-linear spacing
    dxy = pdist2(x, y)

    assert dxy.shape[0] == x.shape[0]
    assert dxy.shape[1] == y.shape[0]

    dxys = sdist.cdist(x, y)
    # max_pdiff = torch.max(torch.abs(dxy.to(float) - torch.tensor(dxys)) /
    #                       torch.abs(torch.tensor(dxys)) )
    assert torch.allclose(dxy.to(float), torch.tensor(dxys), atol=atol, rtol=rtol)

    # test self distance optional arg is consistent with two args
    dx0 = pdist2(x)
    dxx = pdist2(x, x)
    assert torch.allclose(dx0.to(float), torch.tensor(dxx), atol=atol, rtol=rtol)


def test_pdist2_ssim():

    x = torch.arange(0, 10, 2).reshape(-1, 1) * 1.0
    x = x.to(float)
    try:
        ret = pdist2(x, x, distance='ssim')
    except ValueError as err:
        assert(True)
    finally:
        assert(False, "Cannot use scalars with SSIM")

    x = torch.rand((4, 1, 20, 20))
    y = torch.rand((3, 1, 20, 20))
    ret = pdist2(x, y, distance='ssim')
    assert(ret.shape[0] == x.shape[0] and ret.shape[1] == y.shape[0])

    retx0 = pdist2(x, distance='ssim')
    retxx = pdist2(x, x, distance='ssim')
    assert(torch.allclose(retx0, retxx, atol=atol, rtol=rtol))

    # all diags are zero
    assert(torch.allclose(torch.diag(retxx), torch.zeros(x.shape[0])))

    # all off-diags are non-zero (should be true if all x[:] are different)
    assert((retxx.flatten()[1:].view(x.shape[0] - 1, x.shape[0] + 1)[:, :-1].reshape(-1) > 1.e-8).all())


def test_self_knn_expensive():

    k = 4
    nex = 10
    ndim = 2
    x = torch.arange(0, nex * ndim).reshape(nex, ndim) * 1.0
    x = x.to(float)
    x *= x  # square to make it non-linear spacing so result has less symmetry

    dx, dxi = self_knn_expensive(x, k)

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(x)
    dxs, dxsi = nbrs.kneighbors(x)
    dxs = torch.tensor(dxs)
    dxsi = torch.tensor(dxsi)

    assert dx.shape == dxs.shape
    assert torch.allclose(dx, dxs, atol=atol, rtol=rtol)
    assert torch.allclose(dxi, dxsi, atol=atol, rtol=rtol)


def test_knn_expensive():

    k = 4
    nex = 10
    ndim = 2
    x = torch.arange(0, nex * ndim).reshape(nex, ndim) * 1.0
    x = x.to(float)
    x *= x  # square to make it non-linear spacing so result has less symmetry

    y = torch.arange(0, (nex - 2) * ndim).reshape(nex - 2, ndim) * 1.0 + 2.0
    y = y.to(float)
    y = torch.sqrt(y)

    dxy, dxi = knn_expensive(x, y, k)

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(y)
    dxys, dxsi = nbrs.kneighbors(x)
    dxys = torch.tensor(dxys)
    dxsi = torch.tensor(dxsi)

    assert dxy.shape == dxys.shape
    assert torch.allclose(dxy, dxys, atol=atol, rtol=rtol)
    assert torch.allclose(dxi, dxsi, atol=atol, rtol=rtol)


def test_reshape_fortran():
    x = torch.arange(0, 60, 2, dtype=torch.float64).reshape((5,2,3)) * 1.0
    x *= x / 3  # non-equal spacing and not perfect binary

    x2 = reshape_fortran(x, [5, 2, 3])
    x3 = np.reshape(x.numpy(), [5, 2, 3], order='F')
    assert torch.allclose(x2.cpu(), torch.tensor(x3, dtype=torch.float64), atol=0, rtol=0)


def test_get_degen_patt():

    data = laplacian_eig_truth(1)
    dpatt, davg, dstd = get_degen_patt(data)
    assert torch.allclose(torch.tensor(dpatt), torch.tensor([1]))
    data = laplacian_eig_truth(2)
    dpatt, davg, dstd = get_degen_patt(data)
    assert torch.allclose(torch.tensor(dpatt), torch.tensor([1,1]))
    data = laplacian_eig_truth(3)
    dpatt, davg, dstd = get_degen_patt(data)
    assert torch.allclose(torch.tensor(dpatt), torch.tensor([1,2]))
    data = laplacian_eig_truth(6)
    dpatt, davg, dstd = get_degen_patt(data)
    assert torch.allclose(torch.tensor(dpatt), torch.tensor([1,2,2,1]))
