import os, time
import torch
from PySEC.nystrom_cidm import cidm, nystrom #, nystrom_grad, nystrom_gradu
from PySEC.del0 import del0
from PySEC.del1 import del1_as_einsum
from typing import List, Union

class ManifoldApproximators():
    """ wrapper for manifold approximators
    Note: if you define your own manifold approximator the return style needs to be:
    @TODO: match del0 and cidm signatures then document
    """
    def __init__(self, method='del0'):
        if callable(method):
            self.method = method
        else:
            try:
                self.method = getattr(self, method.lower())
            except AttributeError:
                raise ValueError(f'Unknown manifold fit method: {method}')

    def __call__(self, *args, **kwargs):
        return self.distance(*args, **kwargs)

    def cidm(self, *args, **kwargs):
        return cidm(*args, **kwargs)

    def del0(self, *args, **kwargs):
        return del0(*args, **kwargs)


class ManifoldPoints():
    """ Holds the information about some batched points in the eigen basis, with option to embed
    This is done for intermediate storage because mapping to the eigen basis is expensive
    """
    def __init__(self, ubasis: torch.Tensor):
        self.ubasis = ubasis

    def embed(self, embedding: torch.Tensor):
        """ linear map """
        return self.ubasis @ embedding


class Manifold():
    """ Class that will fit data to a manifold, some terminology:
        Ambient: the dimension of a single raw input point, usually the largest dimension
        Intrinsic: possible ground truth dimension of the underlying manifold
        kNN: k nearest neighbors
        Nystrom: an application of the Nytsrom extension to the eigen basis, allowing arbitrary points to be mapped to
            a manifold's eigen basis
    """

    def __init__(self, method: str = 'cidm', k: int = None, **kwargs):
        """ intialize fitting args
        :param method: which ManifoldApproximator
        :param k: number of nearest neighbors
        """
        self.k = k
        self.fit_method = ManifoldApproximators(method)
        self.kwargs = kwargs

    def fit(self, data: torch.Tensor, intrinsic_params: torch.Tensor = None):
        """ fit the data with optional intrinsic parameters
        """
        self.data = data
        self.intrinsic_params = intrinsic_params
        t0 = time.time()
        ret0 = self.fit_method(self.data, self.k, **self.kwargs) #cidm(self.data, )  # k=k, k2=k2, nvars=4*k)
        t1 = time.time()
        print(f'Time for manifold fit: {t1 - t0:.2f}s')
        self.u, self.l, self.peq, self.qest, self.eps, self.dim, self.KP = ret0

        self.Xhat = self.u.T @ torch.diag(self.peq) @ self.data.view(self.data.shape[0], -1).to(self.u)
        self.Shat = None
        if self.intrinsic_params is not None:
            self.Shat = self.u.T @ torch.diag(self.peq) @ self.intrinsic_params

        return self

    def embed(self, x: Union[ManifoldPoints, torch.Tensor], embedding: Union[List, str] = 'ambient'):
        """ Map an abritrary point onto the manfifold and return your desired embedding, default is ambient
        :param x: batched points, a tensor is assumed to be in ambient
        :param embedding: either a string specifying the embedding label, options are 'ambient', 'intrinsic',
            'ubasis', or a linear map of your choice
        :return: a point in the embedding you chose, will be a list of points if embedding is a list
        """

        if isinstance(x, torch.Tensor):
            points = self.map_to_manifold(x)
        else:
            points = x

        if isinstance(embedding, List):
            embed_list = embedding
        else:
            embed_list = [embedding,]

        ret = []
        for emb in embed_list:
            if isinstance(emb, torch.Tensor):
                ret.append(points.embed(emb))
            elif emb.lower().startswith('a'):
                ret.append(points.embed(self.Xhat))
            elif emb.lower().startswith('i'):
                ret.append(points.embed(self.Shat))
            elif emb.lower().startswith('u'):
                ret.append(points.ubasis)

        return ret if isinstance(embedding, List) else ret[0]


    def map_to_manifold(self, x: torch.Tensor):
        """ Map an arbitrary point to the manifold in the manifold basis (Nystrom)
        :param x: batched tensor of points in ambient dimension
        :return: ManifoldPoint
        """
        return ManifoldPoints(nystrom(x, self.KP))


class SEC():
    def __init__(self, manifold: Manifold, basis_size: int = None):
        self.manifold = manifold

    def fit(self):
        self.n1 = min(80, self.KP.k)  # use kNN size capped at 80
        t2 = time.time()
        ret1 = del1_as_einsum(self.u, self.l, torch.diag(self.peq), self.n1)
        t3 = time.time()
        print(f'Time for Del1: {t3 - t2:.2f}s')
        self.u1, self.l1, self.d1, _, self.h1, _ = ret1

