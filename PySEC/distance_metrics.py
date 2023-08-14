import torch
from torchmetrics.functional import structural_similarity_index_measure
from kornia.geometry.transform import build_laplacian_pyramid as blp

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


if __name__ == '__main__':

    dm = DistanceMetric('euclidean')
    x = torch.rand((1, 1, 40, 40))
    y = torch.rand_like(x)
    debug_var = 1
    test = dm(x, y)

    dm = DistanceMetric('euclid')

    dm = DistanceMetric('ssim')
    test = dm(x, y)

    dm = DistanceMetric('lap1')
    test = dm(x, y)

    dm = DistanceMetric()
    test = dm(x, y)

    debug_var = 1

