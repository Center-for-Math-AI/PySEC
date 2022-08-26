import torch
from PySEC.distance_funcs import *
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
    assert torch.allclose(dx.to(float), torch.tensor(dxs), atol=atol, rtol=rtol)


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
