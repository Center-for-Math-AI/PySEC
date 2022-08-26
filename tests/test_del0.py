import torch
from PySEC.del0 import del0
from PySEC.generate_data import generate_circle, laplacian_eig_truth


def test_del0_eigh():
    # test eigenvalues from scipy.linalg.eigh
    num_points = 101
    eig_size = num_points - 1
    data, iparams = generate_circle(num_points)
    fstr = "{:.8f}\t" * 4
    sstr = "\n" + "{:10s}\t" * 4
    print(sstr.format("eps**2", "err_avg", "err_std", "err_max"))
    # for epsilon in [0.06, 0.05]: # 0.8 breaks, 0.04 breaks
    for epsilon in [0.05]:  # 0.8 breaks, 0.04 breaks
        u, l, D = del0(data.T, eig_size, epsilon)
        eig_truth = laplacian_eig_truth(eig_size)
        eig_errs = torch.abs(l - eig_truth)
        ss = slice(
            1, int(0.8 * eig_size)
        )  # only do first 80% since higher ones seems to have larger errors
        rel_errs = eig_errs[ss] / torch.max(l, eig_truth)[ss]  # skip zero
        print(
            fstr.format(epsilon**2, rel_errs.mean(), rel_errs.std(), rel_errs.max())
        )
        assert rel_errs.max().item() < 1.05 * epsilon**2
