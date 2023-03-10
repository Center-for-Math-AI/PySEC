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


def dist(x, y, distance='euclidean'):

    if callable(distance):
        return distance(x, y)

    elif 'euclid' in distance.lower():
        return torch.linalg.norm((x-y).view(x.shape[0], -1), ord=2, dim=1)

    elif 'ssim' in distance.lower():
        return 1. - structural_similarity_index_measure(x, y, reduction='none')

    elif 'lap' in distance.lower():
        max_level = int(torch.ceil(torch.log2(torch.as_tensor(min(x.shape[-1], x.shape[-2])) / 8)))
        xlp = blp(x, max_level=max_level, border_type='reflect', align_corners=False)
        ylp = blp(y, max_level=max_level, border_type='reflect', align_corners=False)

        xlp2 = [xp[:, :, :x.shape[-2] // 2 ** ii, :x.shape[-1] // 2 ** ii] for ii, xp in enumerate(xlp)]
        ylp2 = [yp[:, :, :x.shape[-2] // 2 ** ii, :x.shape[-1] // 2 ** ii] for ii, yp in enumerate(ylp)]

        l1_diffs = torch.cat(
            [torch.linalg.norm((xp - yp).reshape(xp.shape[0], -1), ord=1, dim=1, keepdim=True)
             for ii, (xp, yp) in enumerate(zip(xlp2, ylp2))],
            dim=1)

        lp_weights = 2. ** (2 * torch.arange(max_level, dtype=x.dtype, device=x.device))
        return torch.sum(l1_diffs * lp_weights[None, :], dim=1)

    else:
        return ValueError(f'Unknown distance metric: {distance}')


def pdist2(x, y=None, distance='euclidean', batch_size=128, compute_device=None, progress=False):
    """
    same as above, but for transpose x and handles two args so no just self,
    also is done without creating a 3d object:
    d_xy = sum_s (x_s - y_s)^2 = sum_s ( x_s^2 + y_s^2 - 2*x_s*y_s )
    where the last term can be computed with einsum
    #NB: SSIM specific args
    :param batch_size: batch size for data loading to compute_device
    :param compute_device: torch device for computing, can be different than input tensor device
    :param progress: whether to print a tqdm progress bar to stderr
    :returns: x.shape[0] by y.shape[0] matrix with x's dtype and device
    """

    if compute_device is None:
        compute_device = x.device

    if 'euclid' in distance.lower():

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

    elif 'sim' in distance.lower():
        # from torchmetrics import StructuralSimilarityIndexMeasure
        # ssim = StructuralSimilarityIndexMeasure(data_range=1., reduction='none').to(x.device)
        from torchmetrics.functional import structural_similarity_index_measure
        #NB: SSIM can be negative! use the 1 - SSIM for distance
        # Laplacian pyramids are fast to compute, maybe the euclidean distance of those is better

        if y is None:

            # This would be better if ix1 and ix2 were computed on the fly
            ix1, ix2 = torch.triu_indices(x.shape[0], x.shape[0], 1) # upper triangular, skip diag
            idx_ds = torch.utils.data.TensorDataset(ix1, ix2)
            idx_dl = torch.utils.data.DataLoader(idx_ds, batch_size=batch_size, shuffle=False)

            tmp = torch.empty((len(ix1),), dtype=x.dtype, device=compute_device)

            if progress:
                pbar = trange(len(ix1), unit="Element", ncols=120, position=0, leave=True)
                pbar.set_description(f"pdist2:")

            ioff = 0
            for batch in idx_dl:
                i1, i2 = batch[0], batch[1]
                tmp[ioff:ioff + len(i1)] = 1.0 - structural_similarity_index_measure(
                    x[i1].to(compute_device),
                    x[i2].to(compute_device),
                    reduction='none')
                ioff += len(i1)
                if progress: pbar.update(len(i1))

            # zeros for diag, lets accumulate one triangular into the other
            ret = torch.zeros((x.shape[0], x.shape[0]), dtype=x.dtype, device=x.device)
            ret[ix1, ix2] = tmp.to(ret.device)
            ret = ret + ret.t() # copy upper tri to lower tri
            del ix1, ix2, tmp
            if progress: pbar.close()

        else:

            yds = torch.utils.data.TensorDataset(y)
            ydl = torch.utils.data.DataLoader(yds, batch_size=batch_size, shuffle=False)
            ret = torch.empty((x.shape[0], y.shape[0]), dtype=x.dtype, device=x.device)
            tmp = torch.empty((y.shape[0]), dtype=x.dtype, device=compute_device)
            if progress:
                pbar = trange(len(x)*len(y), unit="Element", ncols=120, position=0, leave=True)
                pbar.set_description(f"pdist2:")

            for ix, xrow in enumerate(x):

                xrow = xrow.to(compute_device)

                ioff = 0
                for batch in ydl:
                    yj = batch[0].to(compute_device)
                    tmp[ioff:ioff+len(yj)] = 1.0 - structural_similarity_index_measure(
                        xrow.unsqueeze(0).expand(yj.shape[0], *xrow.shape),
                        yj, reduction='none')

                    ioff += len(yj)
                    if progress: pbar.update(len(yj))

                ret[ix] = tmp.to(ret.device)

            del tmp
            if progress: pbar.close()



    elif 'lap' in distance.lower():

        if y is None:
            y = x

        yds = torch.utils.data.TensorDataset(y)
        ydl = torch.utils.data.DataLoader(yds, batch_size=batch_size, shuffle=False)
        ret = torch.empty((x.shape[0], y.shape[0]), dtype=x.dtype, device=x.device)
        tmp = torch.empty((y.shape[0]), dtype=x.dtype, device=compute_device)
        if progress:
            pbar = trange(len(x) * len(y), unit="Element", ncols=120, position=0, leave=True)
            pbar.set_description(f"pdist2:")

        for ix, xrow in enumerate(x):

            xrow = xrow.to(compute_device)

            ioff = 0
            for batch in ydl:
                yj = batch[0].to(compute_device)
                tmp[ioff:ioff + len(yj)] = dist(
                    xrow.unsqueeze(0).expand(yj.shape[0], *xrow.shape),
                    yj, distance=distance)

                ioff += len(yj)
                if progress: pbar.update(len(yj))

            ret[ix] = tmp.to(ret.device)

        del tmp
        if progress: pbar.close()


    else:
        raise ValueError('Unknown distance metric')

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




def pdist2_batch(x, y=None, distance='euclidean', batch_size=128, compute_device=None, progress=False):
    """
    same as above, but for transpose x and handles two args so no just self,
    also is done without creating a 3d object:
    d_xy = sum_s (x_s - y_s)^2 = sum_s ( x_s^2 + y_s^2 - 2*x_s*y_s )
    where the last term can be computed with einsum
    #NB: SSIM specific args
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

