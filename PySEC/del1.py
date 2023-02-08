import os, torch
from torch import permute as tp
from PySEC.sec_utils import eig_wrap, repmat, reshape_fortran, outer_sum


def del1_as(u, l, D, n=None):
    """ "
    del1 in anti-symmetric frame
    """

    n0 = u.shape[1]
    if n is None:
        n = n0
    n2 = n * n

    uv = u[:, :n].view(u.shape[0], n, 1, 1)
    cijk = uv.expand(uv.shape[0], n, n, n0)
    Du = torch.matmul(D, u)
    Du = Du.view(Du.shape[0], Du.shape[1], 1, 1).expand(Du.shape[0], Du.shape[1], n, n)
    tmp = cijk * tp(cijk, [0, 2, 1, 3]) * tp(Du, [0, 2, 3, 1])
    cijk = tmp.sum(
        dim=0
    )  # .squeeze() # sum already removes dim in python, unlike matlab

    l1 = l.view(l.shape[0], 1, 1, 1, 1).expand(l.shape[0], n, n, n, n)
    l2 = l[:n].view(n, 1, 1, 1, 1).expand(n, n, n0, n, n)
    h0 = cijk.view(cijk.shape[0], cijk.shape[1], cijk.shape[2], 1, 1).expand(
        cijk.shape[0], cijk.shape[1], cijk.shape[2], n, n
    )
    tmp = tp(h0, [3, 0, 2, 4, 1]) * tp(h0, [0, 3, 2, 1, 4])
    tmp *= tp(l2, [1, 0, 2, 3, 4]) + tp(l2, [3, 1, 2, 4, 0]) - tp(l1, [2, 1, 0, 4, 3])
    h0 = tmp.sum(dim=2)

    H = h0 - tp(h0, [1, 0, 2, 3])
    G = h0 + tp(h0, [1, 0, 3, 2]) - tp(h0, [0, 1, 3, 2]) - tp(h0, [1, 0, 2, 3])
    del h0

    D1 = cijk.view(cijk.shape[0], cijk.shape[1], cijk.shape[2], 1, 1).expand(
        cijk.shape[0], cijk.shape[1], cijk.shape[2], n, n
    )
    lambdas1 = tp(l1, [1, 2, 0, 3, 4]) ** 2
    tmp = (
        l2 + tp(l2, [1, 0, 2, 3, 4]) + tp(l2, [1, 3, 2, 0, 4]) + tp(l2, [1, 3, 2, 4, 0])
    )
    lambdas1 -= tp(l1, [1, 2, 0, 3, 4]) * tmp
    tmp = tp(D1, [1, 4, 2, 0, 3]) * tp(D1, [4, 1, 2, 3, 0])
    tmp -= tp(D1, [4, 1, 2, 0, 3]) * tp(D1, [1, 4, 2, 3, 0])
    tmp *= lambdas1
    D1 = 2 * tmp.sum(dim=2).squeeze()

    D1 = reshape_fortran(D1, (n2, n2))  # reshape fotran order for return
    G = reshape_fortran(G, (n2, n2))
    H = reshape_fortran(H, (n2, n2))
    D1 = 0.5 * (D1 + D1.T)
    G = 0.5 * (G + G.T)

    # D1 is the energy matrix for the 1-Laplacian and G is the Hodge
    # Grammian matrix.  The sum D1+G is the Sobolev H^1 Grammian.  This
    # SVD computes the frame representations of a basis for H^1.
    Ut, St, _ = torch.svd(D1 + G, compute_uv=True)

    # We truncate the basis to remove the very high H^1 energy forms
    # which are spurious due to the redundant frame representation (D1
    # and G should be rank deficient due to redundancy).
    NN = torch.where(St / St[0] < 1.0e-3)[0][0] + 1

    # We then project the 1-Laplacian and Hodge Grammian into the H^1
    tmp = torch.matmul(Ut[:, :NN].T, D1)
    D1proj = torch.matmul(tmp, Ut[:, :NN])
    D1proj = 0.5 * (D1proj + D1proj.T)
    tmp = torch.matmul(Ut[:, :NN].T, G)
    Gproj = torch.matmul(tmp, Ut[:, :NN])
    Gproj = 0.5 * (Gproj + Gproj.T)

    # We can now compute the eigenforms of the 1-Laplacian in this basis
    L, U = eig_wrap(D1proj, Gproj, sort_type="smallest mag")

    # Finally we compute the frame coefficients of the eigenforms
    U = torch.matmul(Ut[:, :NN], U.real)
    return U, L, D1, G, H, cijk
# end def del1_as(u, l, D, n=None):


def del1_as_einsum(u, l, D, n=None):
    # @TODO: D from del0 and cidm are diag matrices, can swith logic output of del0 to
    # put out a diag matrix and switch del1 logic to have a vector D input

    n0 = u.shape[1]
    if n is None:
        n = n0
    n2 = n * n

    ub = repmat(u[:, :n], [n, n0])
    cijk = torch.einsum("ijkl,ikjl,iljk->jkl", ub, ub, repmat(D @ u, [n, n]))

    perms = outer_sum(outer_sum(l[:n], l[:n]), -l)
    h0 = torch.einsum("ljs,ljs,iks->ijkl", perms, cijk, cijk)
    del perms

    H = h0 - tp(h0, [1, 0, 2, 3])
    H = reshape_fortran(H, (n2, n2))  # reshape in fortran order for return

    G = h0 + tp(h0, [1, 0, 3, 2]) - tp(h0, [0, 1, 3, 2]) - tp(h0, [1, 0, 2, 3])
    G = reshape_fortran(G, (n2, n2))  # reshape in fortran order for return
    G = 0.5 * (G + G.T)

    del h0

    perms = outer_sum(l[:n], l[:n])
    perms = outer_sum(perms, perms)  # 4d object of addition permutations
    l2 = 2 * l  # incorporate 2 into the smallest object and let it propagate
    D1 = torch.einsum("s,ils,jks->ijkl", l2, cijk, cijk) - torch.einsum(
        "s,iks,jls->ijkl", l2, cijk, cijk
    )
    D1 *= perms
    del perms
    l2 *= l  # 2 * l^2
    D1 += torch.einsum("s,iks,jls->ijkl", l2, cijk, cijk) - torch.einsum(
        "s,ils,jks->ijkl", l2, cijk, cijk
    )
    del l2
    # @TODO: do some timings and test if instead of doing 4 einsums above, just
    # do 2, then instead of subtracting the permutation subtract a torch.permute

    D1 = reshape_fortran(D1, (n2, n2))  # reshape fotran order for return
    D1 = 0.5 * (D1 + D1.T)

    # D1 is the energy matrix for the 1-Laplacian and G is the Hodge
    # Grammian matrix.  The sum D1+G is the Sobolev H^1 Grammian.  This
    # SVD computes the frame representations of a basis for H^1.
    Ut, St, _ = torch.svd(D1 + G, compute_uv=True)

    # We truncate the basis to remove the very high H^1 energy forms
    # which are spurious due to the redundant frame representation (D1
    # and G should be rank deficient due to redundancy).
    NN = torch.where(St / St[0] < 1.0e-3)[0][0] + 1

    # We then project the 1-Laplacian and Hodge Grammian into the H^1
    D1proj = Ut[:, :NN].T @ D1 @ Ut[:, :NN]
    D1proj = 0.5 * (D1proj + D1proj.T)
    Gproj = Ut[:, :NN].T @ G @ Ut[:, :NN]
    Gproj = 0.5 * (Gproj + Gproj.T)

    # We can now compute the eigenforms of the 1-Laplacian in this basis
    L, U = eig_wrap(D1proj, Gproj, sort_type="smallest mag")

    # Finally we compute the frame coefficients of the eigenforms
    U = torch.matmul(Ut[:, :NN], U.real)
    return U, L, D1, G, H, cijk
# end def del1_as_einsum(u, l, D, n=None):
