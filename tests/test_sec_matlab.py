import numpy as np
from PySEC import del0, del1
import scipy.linalg
import os, torch
import pytest
from PySEC.del0 import del0
from PySEC.del1 import del1_as, del1_as_einsum
from PySEC.generate_data import *
from PySEC.sec_utils import compare_eigenpairs

try:
    import matlab
    import matlab.engine as me
except ModuleNotFoundError as err:
    print("## CANNOT FIND MATLAB, if you have matlab try to install engine: ##")
    print("mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html")
    pytest.exit("Skipping matlab tests, cannot find matlab")

try:
    eng = me.start_matlab()
    eng.isprime(3)  # test call
except Exception as err:  # what exception will this be?
    print("## CANNOT START MATLAB ##")
    pytest.exit("Skipping matlab tests, cannot start matlab, do you have a license?")


sec_path = os.path.join(os.path.split(os.path.dirname(__file__))[0], "matlab")
eng.addpath(sec_path)


torch.set_default_dtype(torch.float64)


def compare_to_matlab_Del1_AS(
    pyfunc, data, eig_size, sec_size, epsilon, atol=1.0e-8, rtol=1.0e-6, eig_tol=1.0e2
):

    mdata = matlab.double(data.detach().numpy())
    # u, l, D = del0(data, eig_size, epsilon)  # python implementation
    um, lm, Dm = eng.Del0(mdata, float(eig_size), epsilon, nargout=3)
    um, lm, Dm = eng.real(um), eng.real(lm), eng.real(Dm)  # force cast to real
    umt = torch.as_tensor(um, dtype=float)
    lmt = torch.as_tensor(lm, dtype=float)
    Dmt = torch.as_tensor(Dm, dtype=float)

    retm = eng.Del1AS(um, lm, Dm, sec_size, nargout=6)
    retp = pyfunc(umt, lmt.squeeze(), Dmt, sec_size)

    # print(f'passed {pyfunc.__name__} @eps {epsilon} returns: ', end='')

    same = [
        None,
    ] * 6
    if eig_tol is not None:  # compare eigen pairs
        lmt = torch.as_tensor(retm[1], dtype=float).squeeze()
        umt = torch.as_tensor(retm[0], dtype=float).squeeze().T
        lpt = torch.as_tensor(retp[1], dtype=float).squeeze()
        upt = torch.as_tensor(retp[0], dtype=float).squeeze().T
        same[0] = compare_eigenpairs(
            lmt, umt, lpt, upt, atol=atol * eig_tol, rtol=rtol * eig_tol
        )
        assert same[0], "Eigen pairs are too different"

    for ir in range(1, 6):  # skip eigenvecs since not unique
        rm = torch.as_tensor(retm[ir], dtype=float).squeeze()
        rp = torch.as_tensor(retp[ir], dtype=float).squeeze()
        assert (
            rm.shape == rp.shape
        ), f"Shape mismatch in return arg({ir}), matlab: {rm.shape}, python:{rp.shape}"
        same[ir] = torch.allclose(rm, rp, atol=atol, rtol=rtol)

    check_same = same[1:] if eig_tol is None else same
    if not all(check_same):
        print("\nReturn matches: ", end="")
        print(same)
    assert all(check_same)


def test_del0_circle():
    num_points = 120
    eig_size = 100
    data, iparams = generate_circle(num_points)
    mdata = matlab.double(data.detach().numpy())

    epsilon = 0.05
    retm = eng.Del0(mdata, eig_size, epsilon, nargout=3)
    retp = del0(data.T, eig_size, epsilon)

    # compare eigen pairs
    lmt = torch.as_tensor(retm[1], dtype=float).squeeze()
    umt = torch.as_tensor(retm[0], dtype=float).squeeze().T
    lpt = torch.as_tensor(retp[1], dtype=float).squeeze()
    upt = torch.as_tensor(retp[0], dtype=float).squeeze().T
    same = compare_eigenpairs(lmt, umt, lpt, upt, atol=1.0e-8, rtol=1.0e-6)
    assert same, "Eigen pairs are too different"

    for ir in range(1, 3):
        rm = torch.as_tensor(retm[ir], dtype=float).squeeze()
        rp = torch.as_tensor(retp[ir], dtype=float).squeeze()
        assert (
            rm.shape == rp.shape
        ), f"Shape mismatch in return arg({ir}), matlab: {rm.shape}, python:{rp.shape}"
        same = torch.allclose(rm, rp, atol=1.0e-8, rtol=1.0e-6)
        if not same:
            print()
            print(same)  # break here
        assert same, f"value mismatch in return arg {ir}"


def test_del0_torus():
    num_points = 2 * 10**3
    eig_size = 100
    data, iparams = generate_torus(num_points)
    mdata = matlab.double(data.detach().numpy())
    epsilon = 0.2

    retm = eng.Del0(mdata, eig_size, epsilon, nargout=3)
    retp = del0(data.T, eig_size, epsilon)

    # compare eigen pairs
    lmt = torch.as_tensor(retm[1], dtype=float).squeeze()
    umt = torch.as_tensor(retm[0], dtype=float).squeeze().T
    lpt = torch.as_tensor(retp[1], dtype=float).squeeze()
    upt = torch.as_tensor(retp[0], dtype=float).squeeze().T
    same = compare_eigenpairs(lmt, umt, lpt, upt, atol=1.0e-8, rtol=1.0e-6)
    assert same, "Eigen pairs are too different"

    for ir in range(1, 3):
        rm = torch.as_tensor(retm[ir], dtype=float).squeeze()
        rp = torch.as_tensor(retp[ir], dtype=float).squeeze()
        assert (
            rm.shape == rp.shape
        ), f"Shape mismatch in return arg({ir}), matlab: {rm.shape}, python:{rp.shape}"
        same = torch.allclose(rm, rp, atol=1.0e-8, rtol=1.0e-6)
        if not same:
            print()
            print(same)  # break here
        assert same, f"value mismatch in return arg {ir}"


def test_del1_as_circle():
    """ Note: doesn't pass after Del1 with pseudo inv sqrt gen eig """
    num_points = 101
    eig_size = 100
    sec_size = 20
    data, iparams = generate_circle(num_points)
    epsilon = 0.05

    pyfuncs = [del1_as, del1_as_einsum]
    for func in pyfuncs:
        compare_to_matlab_Del1_AS(func, data, eig_size, sec_size, epsilon)


def test_del1_as_torus():
    """ Note: doesn't pass after Del1 with pseudo inv sqrt gen eig """
    num_points = 5 * 10**3
    eig_size = 100
    sec_size = 20
    data, iparams = generate_torus(num_points)
    epsilon = 0.2

    pyfuncs = [del1_as, del1_as_einsum]
    for func in pyfuncs:
        compare_to_matlab_Del1_AS(
            func, data, eig_size, sec_size, epsilon, eig_tol=None
        )  # dont' check eigenvectors
