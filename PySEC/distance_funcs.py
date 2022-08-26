import torch

"""
Notes: (knn == k nearest neighbor algo)
- I found torch.cdist to be non-deterministic, which is horrible, so I created the
    self_pair_dist_p2 function below
- scipy has a cdist function, but it doesn't do knn
- scikit has knn in:
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    which is really nice because it uses a tree, so n*log(n) scaling, but it's on the cpu
- the faiss package does knn on the gpu with a bit of boiler plate
    https://machinelearningapplied.com/fast-gpu-based-nearest-neighbors-with-faiss/
"""


def self_pair_dist_p2(x):
    """
    pariswise distance with yourself, p2 norm for distance
    NB: if you pass in a torch.float32 it will give results bad enough to fail test_del0.py
    :param x: data, shape == [ambient, num_examples]
    :return: distances of shape == [num_examples, num_examples]
    """
    adim = x.shape[0]  # ambient dim
    nex = x.shape[1]  # num examples
    # combos = nex * (nex-1) / 2 # triangular matrix size
    # dmat_full = torch.empty((nex, nex, adim), dtype=x.dtype)
    # @TODO: .view() is easier to read than .unsqueeze()
    x_rep = x.unsqueeze(dim=-1).expand(x.shape[0], x.shape[1], nex).moveaxis(0, -1)
    dmat_full = x_rep - torch.moveaxis(x_rep, 1, 0)
    return torch.sqrt((dmat_full * dmat_full).sum(dim=-1))


def pdist2(x, y=None):
    """
    same as above, but for transpose x and handles two args so no just self,
    also is done without creating a 3d object:
    d_xy = sum_s (x_s - y_s)^2 = sum_s ( x_s^2 + y_s^2 - 2*x_s*y_s )
    where the last term can be computed with einsum
    """
    x2 = torch.sum(x * x, dim=-1)
    # @TODO: test it's the same
    # x2 = torch.einsum('ij,ij->i', x, x) no intermediate object, test first
    if y is None:
        ret = x2[:, None] + x2 + torch.einsum("is,js->ij", -2 * x, x)
    else:
        if y.shape[-1] != x.shape[-1]:
            raise ValueError("ERR(pdist2): shape mismatch")
        y2 = torch.sum(y * y, dim=-1)
        ret = x2[:, None] + y2 + torch.einsum("is,js->ij", -2 * x, y)

    # truncate numerically small distances and negatives from expanding the square
    ret[ret < 1.0e-10] = 0.0
    return torch.sqrt(ret)


def self_knn_expensive(x, k):
    # also can explore using torch.nn.PairwiseDistance(p=2, eps=1.e-20)
    # for when we exceed gpu memory, but should probably switch to a
    # tree algo before that point
    dx = pdist2(x)
    kn = min(k, dx.shape[0])
    return torch.topk(dx, kn, largest=False)


def knn_expensive(x, y, k):
    dx = pdist2(x, y)
    kn = min(k, dx.shape[1])
    return torch.topk(dx, kn, largest=False)
