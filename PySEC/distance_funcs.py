import torch
from tqdm import tqdm, trange
from torchmetrics.functional import structural_similarity_index_measure
from kornia.geometry.transform import build_laplacian_pyramid as blp


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


class DistanceMetric:
    """ Wrapper class for distance metrics
    """

    def __init__(self, distance='euclidean'):
        if callable(distance):
            self.distance = distance
        else:
            try:
                self.distance = getattr(self, distance.lower())
            except AttributeError:
                raise ValueError(f'Unknown distance function: {distance}')

    def __call__(self, *args, **kwargs):
        return self.distance(*args, **kwargs)

    def euclidean(self, x, y):
        return torch.linalg.norm((x - y).view(x.shape[0], -1), ord=2, dim=1)

    def ssim(self, x, y, **kwargs):
        return 1. - structural_similarity_index_measure(x, y, reduction='none', **kwargs)

    def lap1(self, x, y, level_weights=None):

        max_level = int(torch.ceil(torch.log2(torch.as_tensor(min(x.shape[-1], x.shape[-2])) / 8)))
        xlp = blp(x, max_level=max_level, border_type='reflect', align_corners=False)
        ylp = blp(y, max_level=max_level, border_type='reflect', align_corners=False)

        xlp2 = [xp[:, :, :x.shape[-2] // 2 ** ii, :x.shape[-1] // 2 ** ii] for ii, xp in enumerate(xlp)]
        ylp2 = [yp[:, :, :x.shape[-2] // 2 ** ii, :x.shape[-1] // 2 ** ii] for ii, yp in enumerate(ylp)]

        l1_diffs = torch.cat(
            [torch.linalg.norm((xp - yp).reshape(xp.shape[0], -1), ord=1, dim=1, keepdim=True)
             for ii, (xp, yp) in enumerate(zip(xlp2, ylp2))],
            dim=1)

        if level_weights is None:
            lp_weights = 2. ** (2 * torch.arange(max_level, dtype=x.dtype, device=x.device))
        else:
            lp_weights = level_weights
        return torch.sum(l1_diffs * lp_weights[None, :], dim=1)

    def lapeuc(self, x, y, euclid_weight=0.5, level_weights=None):
        if euclid_weight < 0.0 or euclid_weight > 1.0:
            raise ValueError(f'Relative euclid_weigtht must be in range [0, 1]')
        lap1_weight = 1.0 - euclid_weight
        return 2. * (euclid_weight * self.euclidean(x, y) + lap1_weight * self.lap1(x, y, level_weights))


def pdist2(x, y=None, distance='euclidean', batch_size=128, compute_device=None, progress=False):
    """ Pairwise distance function, batched
    :param x: data with shape (N, C, H, W), or (N, D), such that first dim is number of points
    :param y: (optional) if not provided returns distance of x with itself, otherwise distance between x & y
    :param batch_size: batch size for data loading to compute_device
    :param compute_device: torch device for computing, can be different than input tensor device
    :param progress: whether to print a tqdm progress bar to stderr
    :returns: x.shape[0] by y.shape[0] matrix with x's dtype and device
    """

    # @TODO: see if it's faster to generate indices on the fly, will be necessary for massive datasets
    # reading tmp into ret will be more of pain...

    if compute_device is None:
        compute_device = x.device

    if y is None:

        symm_flag = True
        y = x
        iix, iiy = torch.triu_indices(x.shape[0], x.shape[0], 1) # upper triangular, skip diag
        idx_ds = torch.utils.data.TensorDataset(iix, iiy)
        idx_dl = torch.utils.data.DataLoader(idx_ds, batch_size=batch_size, shuffle=False)

    else:

        symm_flag = False
        iix, iiy = torch.meshgrid(torch.arange(x.shape[0]), torch.arange(y.shape[0]), indexing='ij')  #
        iix, iiy = iix.ravel(), iiy.ravel()
        idx_ds = torch.utils.data.TensorDataset(iix, iiy)
        idx_dl = torch.utils.data.DataLoader(idx_ds, batch_size=batch_size, shuffle=False)

    tmp = torch.empty((len(iix),), dtype=x.dtype, device=compute_device)
    dist_metric = DistanceMetric(distance=distance)

    if progress:
        pbar = trange(len(iix), unit="Element", ncols=120, position=0, leave=True)
        pbar.set_description(f"pdist2 ({distance[:6]}): ")

    ioff = 0
    for batch in idx_dl:
        i1, i2 = batch[0], batch[1]
        xb, yb = x[i1].to(compute_device), y[i2].to(compute_device)
        tmp[ioff:ioff + len(i1)] = dist_metric(xb, yb)
        ioff += len(i1)
        if progress: pbar.update(len(i1))

    if symm_flag: # x == y
        # zeros for diag, lets accumulate one triangular into the other
        ret = torch.zeros((x.shape[0], x.shape[0]), dtype=x.dtype, device=x.device)
        ret[iix, iiy] = tmp.to(ret.device)
        ret = ret + ret.t() # copy upper tri to lower tri

    else:
        ret = torch.empty((x.shape[0], y.shape[0]), dtype=x.dtype, device=x.device)
        ret[iix, iiy] = tmp.to(ret.device)

    if progress: pbar.close()
    return ret


def self_knn_expensive(x, k, **dist_args):
    # also can explore using torch.nn.PairwiseDistance(p=2, eps=1.e-20)
    # for when we exceed gpu memory, but should probably switch to a
    # tree algo before that point
    dx = pdist2(x, **dist_args)
    kn = min(k, dx.shape[0])
    return torch.topk(dx, kn, largest=False)


def knn_expensive(x, y, k, **dist_args):
    dx = pdist2(x, y, **dist_args)
    kn = min(k, dx.shape[1])
    return torch.topk(dx, kn, largest=False)



if __name__ == '__main__':

    mydist = 'euclidean'
    dm = DistanceMetric()
    x = torch.rand((10, 1, 40, 40))
    y = torch.rand_like(x)
    debug_var = 1
    # t0 = dist(x, y, mydist)
    t1 = dm(x, y)
    # torch.allclose(t0, t1, atol=1e-10, rtol=1e-10)

    debug_var = 1

