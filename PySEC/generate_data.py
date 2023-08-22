import torch
from math import pi, floor, sqrt
from PySEC.sec_utils import repmat


def generate_circle(n):
    # generate points on a circle in xy plane return xyz
    theta = 2.0 * pi * torch.linspace(1.0 / n, 1, n, dtype=float)  # force float64
    intrinsic = torch.empty((2, n), dtype=theta.dtype)
    intrinsic[0] = torch.cos(theta)
    intrinsic[1] = torch.sin(theta)
    data = torch.zeros((3, n), dtype=theta.dtype)
    data[:2] = intrinsic
    return data, intrinsic


def generate_noisy_circle(n, std):
    data, intrinsic = generate_circle(n)
    intrinsic += torch.normal(0.0, std * 0.1, intrinsic.shape)
    data += torch.normal(0.0, std, data.shape)
    return data, intrinsic


def generate_torus(n: int, r_out: float = 2, r_inn: float = 1, noise=0):
    """
    :param n: number of points total
    :param r_out: outer radius
    :param r_inn: inner radius
    :param noise: noise of the grid point to break uniformity
    """
    nn = floor(sqrt(n))
    t = torch.linspace(1.0 / nn, 1.0, nn, dtype=float) * 2.0 * pi
    nn = len(t)
    phi = repmat(t, [nn])
    theta = phi.detach().clone().T.reshape(-1)
    phi = phi.reshape(-1)

    if noise > 0:
        theta += torch.randn(*theta.shape) * noise
        phi += torch.randn(*theta.shape) * noise

    intrinsic = torch.stack((theta, phi))
    x = (r_out + r_inn * torch.cos(theta)) * torch.cos(phi)
    y = (r_out + r_inn * torch.cos(theta)) * torch.sin(phi)
    z = r_inn * torch.sin(theta)
    data = torch.stack((x, y, z))
    return data, intrinsic
# end def generate_torus(n):


def generate_double_torus(n: int, r_out: float = 2, r_inn: float = 1):
    """
    :param n: number of points total
    :param r_out: outer radius
    :param r_inn: inner radius
    """

    nn = floor(sqrt(n / 2))
    td, ti = generate_torus(n / 2, r_out, r_inn)
    x = td[0]
    y = td[1]
    z = td[2]
    theta = ti[0]
    phi = ti[1]

    r_cut = r_out  # r_inn + 0.75*(r_out - r_inn)
    x_lt_ro = torch.where(x < r_cut)
    x_gt_ro = torch.where(x > -r_cut)

    theta_ret = torch.cat((theta[x_lt_ro].clone() - 2.0 * pi, theta[x_gt_ro]))
    phi_ret = torch.cat((phi[x_lt_ro], phi[x_gt_ro]))

    intrinsic = torch.empty((2, theta_ret.shape[0]), dtype=theta.dtype)
    data = torch.empty((3, theta_ret.shape[0]), dtype=theta.dtype)
    intrinsic[0] = theta_ret
    intrinsic[1] = phi_ret

    r_move = r_cut + 0.25 * pi / nn
    data[0] = torch.cat((x[x_lt_ro].clone() - r_move, x[x_gt_ro] + r_move))
    data[1] = torch.cat((y[x_lt_ro], y[x_gt_ro]))
    data[2] = torch.cat((z[x_lt_ro], z[x_gt_ro]))

    return data, intrinsic
# end def generate_double_torus(n):


def laplacian_eig_truth(n):
    # truth values for eigenvalues of the laplacian
    ret = torch.arange(1, (n + 2) // 2, dtype=float) ** 2
    ret = torch.stack((ret, ret), dim=1).view(-1)
    return torch.cat((torch.zeros(1), ret))[:n]

def klein_bottle_3d(u, v):
    """
    http://paulbourke.net/geometry/klein/
    :param u,v: meshgrids
    """
    pi = torch.pi
    half = (0 <= u) & (u < pi)
    r = 4*(1 - torch.cos(u)/2)
    x = 6*torch.cos(u)*(1 + torch.sin(u)) + r*torch.cos(v + pi)
    x[half] = (
        (6*torch.cos(u)*(1 + torch.sin(u)) + r*torch.cos(u)*torch.cos(v))[half])
    y = 16 * torch.sin(u)
    y[half] = (16*torch.sin(u) + r*torch.sin(u)*torch.cos(v))[half]
    z = r * torch.sin(v)
    return x, y, z
