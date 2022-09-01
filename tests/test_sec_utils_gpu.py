import pytest
import torch
import numpy as np
from math import sqrt
from PySEC.distance_funcs import pdist2, self_knn_expensive, knn_expensive
from PySEC.generate_data import laplacian_eig_truth
from PySEC.sec_utils import (reshape_fortran, get_degen_patt,
                             estimate_dimension)
from sklearn.neighbors import NearestNeighbors


atol = 1.0e-14
rtol = 1.0e-12

if not torch.cuda.is_available():
    print('cannot run GPU tests without cuda')
    pytest.exit('cannot run GPU tests without cuda')

dtypes = [torch.float32, torch.float64]
dtype_tols = [[sqrt(atol), sqrt(rtol)], [atol, rtol]]
devices = [f'cuda:{d}' for d in range(torch.cuda.device_count())]

def test_gpu_properties():
    print('\nRunning tests on these GPUs:')
    for device in devices:
        mem = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
        print(f'{device} ({mem:.1f} GB) := {torch.cuda.get_device_name(device)}')
    assert True


## test distance_funcs

def test_pdist2_self_gpu():
    from PySEC.distance_funcs import pdist2
    import scipy.spatial.distance as sdist

    for device in devices:
        for dtype, [atol, rtol] in zip(dtypes, dtype_tols): # this one breaks for float32...

            x = torch.arange(0, 10, 2, dtype=dtype).reshape(-1, 1) * 1.0
            x *= x / 3  # non-equal spacing and not perfect binary
            dxg = pdist2(x.to(device))
            dx = dxg.cpu()

            assert dx.shape[0] == dx.shape[1] == x.shape[0]
            assert torch.allclose(x[:, 0], dx[0], atol=atol, rtol=rtol)

            dxs = sdist.cdist(x, x)
            assert torch.allclose(dx, torch.tensor(dxs, dtype=dtype), atol=atol, rtol=rtol)


def test_pdist2_xy_gpu():
    from PySEC.distance_funcs import pdist2
    import scipy.spatial.distance as sdist

    for device in devices:
        for dtype, [atol, rtol] in zip(dtypes[1:], dtype_tols[1:]): # this one breaks for float32...

            x = torch.arange(0, 10, 2, dtype=dtype).reshape(-1, 1) * 1.0
            x *= x / 3  # non-equal spacing and not perfect binary
            y = torch.arange(0, 8, dtype=dtype).reshape(-1, 1) * 1.0 + 2.0
            y = torch.sqrt(y / 5)

            dxyg = pdist2(x.to(device), y.to(device))
            dxy = dxyg.cpu()

            assert dxy.shape[0] == x.shape[0]
            assert dxy.shape[1] == y.shape[0]

            dxys = sdist.cdist(x.numpy(), y.numpy())
            assert torch.allclose(dxy, torch.tensor(dxys, dtype=dtype), atol=atol, rtol=rtol)

def test_self_knn_expensive():

    k = 4
    nex = 10
    ndim = 2

    for device in devices:
        for dtype, [atol, rtol] in zip(dtypes[0:], dtype_tols[0:]): #

            x = torch.arange(0, nex * ndim, dtype=dtype).reshape(nex, ndim) * 1.0
            x *= x / 3  # non-equal spacing and not perfect binary

            dxg, dxig = self_knn_expensive(x.to(device), k)
            dx, dxi = dxg.cpu(), dxig.cpu()

            xn = x.numpy()
            nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(xn)
            dxs, dxsi = nbrs.kneighbors(xn)
            dxs = torch.tensor(dxs, dtype=dtype)
            dxsi = torch.tensor(dxsi, dtype=torch.int64)

            assert dx.shape == dxs.shape
            assert torch.allclose(dx, dxs, atol=atol, rtol=rtol)
            assert torch.allclose(dxi, dxsi, atol=atol, rtol=rtol)


def test_knn_expensive():

    k = 4
    nex = 10
    ndim = 2

    for device in devices:
        for dtype, [atol, rtol] in zip(dtypes[0:], dtype_tols[0:]): #

            x = torch.arange(0, nex * ndim, dtype=dtype).reshape(nex, ndim) * 1.0
            x *= x / 3  # non-equal spacing and not perfect binary
            y = torch.arange(0, (nex - 2) * ndim, dtype=dtype).reshape(nex - 2, ndim) * 1.0 + 2.0
            y = torch.sqrt(y / 5)

            dxg, dxig = knn_expensive(x.to(device), y.to(device), k)
            dx, dxi = dxg.cpu(), dxig.cpu()

            nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(y.numpy())
            dxs, dxsi = nbrs.kneighbors(x.numpy())
            dxs = torch.tensor(dxs, dtype=dtype)
            dxsi = torch.tensor(dxsi, dtype=torch.int64)

            assert dx.shape == dxs.shape
            assert torch.allclose(dx, dxs, atol=atol, rtol=rtol)
            assert torch.allclose(dxi, dxsi, atol=atol, rtol=rtol)


## test sec_utils

def test_reshape_fortran_gpu():
    for device in devices:
        for dtype, [atol, rtol] in zip(dtypes[0:], dtype_tols[0:]): #

            x = torch.arange(0, 60, 2, dtype=dtype).reshape((5,2,3)) * 1.0
            x *= x / 3  # non-equal spacing and not perfect binary

            x2 = reshape_fortran(x.to(device), [5, 2, 3])
            x3 = np.reshape(x.numpy(), [5, 2, 3], order='F')
            assert torch.allclose(x2.cpu(), torch.tensor(x3, dtype=dtype), atol=0, rtol=0)



def test_estimate_dimension_gpu():
    ''' simply test if it runs on GPU '''
    npoints = 60
    nskip = 2
    shape2 = (10, 3)
    cpu_truth = [[0.30278700590133667,
                  -11.618144035339355],
                 [0.3027215830881817,
                  0.10562932936232935]] # 'ground truth' from numpy/cpu test
    for device in devices:
        for ii, dtype in enumerate(dtypes[0:]):
            #atol, rtol = dtype_tols[ii] # too tight for ddim
            x = torch.arange(0, npoints, nskip, dtype=dtype).reshape(shape2) * 1.0
            x *= x / 3  # non-equal spacing and not perfect binary
            dim, ddim = estimate_dimension(x.to(device), 1.)
            assert torch.allclose(dim, torch.tensor(cpu_truth[ii][0], dtype=torch.float64))
            assert torch.allclose(ddim, torch.tensor(cpu_truth[ii][1], dtype=torch.float64))
            #print(f'\n{dim}\n{ddim}')
