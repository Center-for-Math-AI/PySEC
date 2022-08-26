import numpy as np
import os, torch
from math import ceil, log
import pytest
from PySEC.nystrom_cidm import cidm, nystrom
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
atol = 1.0e-12  # numerical tolerances
rtol = 1.0e-10


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
            new_val = matlab.doulbe(new_val)

        ret[key] = new_val

    return ret


def compare_cidm_matlab(
    data, atol=1.0e-8, rtol=1.0e-6, eig_tol=1.0e0, eig_pairs=None, **kwargs
):
    mdata = matlab.double(data.detach().numpy().T)
    # u, l, peq, qest, eps, dim, kp = cidm(data.T)
    kwkeys = ["nvars", "k", "k2"]
    if len(kwargs) > 0:
        retp = cidm(data, **kwargs)
        try:
            margs = [
                kwargs[key] for key in kwkeys
            ]  # fails for partials, but so could eng.CIDM
        except KeyError as err:
            print(f"ERR: cannot run with partial kwargs: {kwkeys} for matlab CIDM")
            return False
        retm = eng.CIDM(
            mdata, *margs, nargout=6
        )  # engine does not support sparse type, can't return KP
    else:
        retp = cidm(data)
        retm = eng.CIDM(
            mdata, nargout=6
        )  # engine does not support sparse type, can't return KP

    same = [
        None,
    ] * 6
    if eig_tol is not None:
        lmt = torch.as_tensor(retm[1], dtype=float).squeeze()
        umt = torch.as_tensor(retm[0], dtype=float).squeeze().T
        lpt = torch.as_tensor(retp[1], dtype=float).squeeze()
        upt = torch.as_tensor(retp[0], dtype=float).squeeze().T
        ss = slice(None, eig_pairs)
        same[0] = compare_eigenpairs(
            lmt[ss], umt[ss], lpt[ss], upt[ss], atol=atol * eig_tol, rtol=rtol * eig_tol
        )
        assert same[0], "Eigen pairs are too different"

    for ir in range(1, 6):
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


def test_cidm_circle_eigh():
    num_points = 120
    data, iparams = generate_circle(num_points)
    compare_cidm_matlab(data.T, atol=atol, rtol=rtol)


@pytest.mark.filterwarnings(
    "ignore:.*triangular_solve.*:UserWarning"
)  # as per @lfolsom
def test_cidm_circle_lobpcg():
    num_points = 180
    data, iparams = generate_circle(num_points)
    compare_cidm_matlab(data.T, atol=atol, rtol=rtol)


def test_cidm_torus_eigh():
    num_points = 4 * 10**3
    data, iparams = generate_torus(
        num_points, noise=1.0e-4
    )  # noise the lattice to break degeneracies
    sub_idx = torch.arange(0, data.shape[1], 2**4)
    subset = data[:, (sub_idx)].clone()
    k_max = ceil(log(data.shape[1]) ** 2)
    k2_max = ceil(log(data.shape[1]))
    nvars_max = 2 * k_max
    compare_cidm_matlab(
        subset.T,
        atol=atol,
        rtol=rtol,
        eig_tol=1.0e0,
        nvars=nvars_max,
        k=k_max,
        k2=k2_max,
    )  # passes eig_compare after abs(log)


@pytest.mark.filterwarnings(
    "ignore:.*triangular_solve.*:UserWarning"
)  # as per @lfolsom
def test_cidm_torus_lobpcg():
    num_points = 2 * 10**3
    data, iparams = generate_torus(num_points, noise=1.0e-4)
    compare_cidm_matlab(
        data.T, atol=atol, rtol=rtol, eig_tol=1.0e0
    )  # passes eig_compare after abs(log)


def test_nystrom_circle():
    num_points = 120
    data, iparams = generate_circle(num_points)
    u, l, peq, qest, eps, dim, KP = cidm(data.T)

    x2 = data[:, :: num_points // 4].T
    retp = nystrom(x2, KP)
    retm = eng.NystromCIDM(
        matlab.double(x2.clone().numpy().T),
        eng.struct(dataclass2matlab_dict(KP)),
        nargout=3,
    )

    for rp, rm in zip(retp, retm):
        assert torch.allclose(
            rp.view(-1), torch.as_tensor(rm).view(-1), atol=atol, rtol=rtol
        )


def test_nystrom_torus():
    num_points = 1 * 10**3
    data, iparams = generate_torus(
        num_points, noise=1.0e-4
    )  # noise the lattice to break degeneracies
    u, l, peq, qest, eps, dim, KP = cidm(data.T)

    x2 = data[:, :: num_points // 4].T
    retp = nystrom(x2, KP)
    retm = eng.NystromCIDM(
        matlab.double(x2.clone().numpy().T),
        eng.struct(dataclass2matlab_dict(KP)),
        nargout=3,
    )

    for rp, rm in zip(retp, retm):
        assert torch.allclose(
            rp.view(-1), torch.as_tensor(rm).view(-1), atol=atol, rtol=rtol
        )


# @TODO: nystrom a chip and make sure you get the same chip back


# def test_cidm_convergence():
def run_cidm_convergence():

    num_points = 4 * 10**3
    data, iparams = generate_torus(
        num_points, noise=1.0e-4
    )  # noise the lattice to break degeneracies
    debug = True

    k_max = ceil(log(data.shape[1]) ** 2) // 2
    k2_max = ceil(log(data.shape[1]))
    nvars_max = 2 * k_max

    l0 = torch.zeros(2 * k_max)
    u0 = torch.zeros((data.shape[1], 2 * k_max))
    l1 = torch.zeros_like(l0)
    u1 = torch.zeros_like(u0)

    frac_list = torch.linspace(0, 1, 9)
    frac_list[0] = frac_list[1] * 0.5
    for frac in frac_list:
        # linspace is a uniform subsample that looks like rings, np.choice is random sample
        sub_idx = (
            torch.linspace(0, data.shape[1] - 1, round(data.shape[1] * frac.item()))
            .round()
            .to(int)
        )
        # sub_idx = np.random.choice(
        #    data.shape[1], round(data.shape[1] * frac.item()), replace=False
        # )
        subset = data[:, (sub_idx)].clone()

        if debug:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.scatter(subset[0, :], subset[1, :], subset[2, :], marker="o")
            plt.show()

        if subset.shape[-1] < nvars_max:
            continue  # can't do that
        u, l, peq, qest, eps, dim, KP = cidm(subset.T, nvars_max, k_max, k2_max)

        if debug:
            mdata = matlab.double(subset.detach().numpy())
            retm = eng.CIDM(
                mdata, nvars_max, k_max, k2_max, nargout=7
            )  # engine does not support sparse type, can't return KP
            um = torch.as_tensor(retm[0])
            lm = torch.as_tensor(retm[1]).squeeze()
            dm = torch.as_tensor(retm[-1]["d"])
            msame = compare_eigenpairs(l, u.T, lm, um.T)

        u1[: u.shape[0], : u.shape[1]] = u
        l1[: l.shape[0]] = l

        s0 = l.shape[0]
        same = compare_eigenpairs(l0[:s0], u0[:, :s0], l1[:s0], u1[:, :s0])

        l0 = l1.clone()  # update previous step
        u0 = u0.clone()
        print(f"frac: {frac:.3f}, l.shape: {l.shape}, u.shape: {u.shape}, same: {same}")
        print(l[:6])
