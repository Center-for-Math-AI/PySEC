import os, time
import torch
from PySEC.nystrom_cidm import cidm, nystrom #, nystrom_grad, nystrom_gradu
from PySEC.del0 import del0
from PySEC.del1 import del1_as_einsum
from typing import List, Union

class ManifoldMethods():
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
        return self.method(*args, **kwargs)

    def cidm(self, *args, **kwargs):
        return cidm(*args, **kwargs)

    def del0(self, *args, **kwargs):
        return del0(*args, **kwargs)


class ManifoldPoints():
    """ Holds the information about some batched points in a manifold's eigen basis, with option to embed
    This is done for intermediate storage because mapping to the eigen basis is expensive
    """
    def __init__(self, ubasis: torch.Tensor):
        self.ubasis = ubasis

    def embed(self, embedding: torch.Tensor):
        """ linear map
        :param embedding: shape == [ubasis, embedding_dim], ubasis is the size of Manifold.l
        """
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
        self.fit_method = ManifoldMethods(method)
        self.kwargs = kwargs

    def fit(self, data: torch.Tensor, intrinsic_params: torch.Tensor = None):
        """ fit the data with optional intrinsic parameters
        """
        self.data = data
        self.intrinsic_params = intrinsic_params
        t0 = time.time()
        ret0 = self.fit_method(self.data, self.k, **self.kwargs) #cidm(self.data, )  # k=k, k2=k2, nvars=4*k)
        t1 = time.time()
        print(f'Time taken to fit manifold: {t1 - t0:.2f}s')
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
            'ubasis', or a linear map of your choice first dimension of len(self.KP.l)
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
                ret.append(points.embed(self.Xhat).reshape((-1, *self.KP.X.shape[1:])))
            elif emb.lower().startswith('i'):
                ret.append(points.embed(self.Shat))
            elif emb.lower().startswith('u'):
                ret.append(points.ubasis)

        return ret if isinstance(embedding, List) else ret[0]


    def map_to_manifold(self, x: torch.Tensor):
        """ Map an arbitrary point to the manifold in the manifold basis (Nystrom)
        The point of the ManifoldPoints class is expose this layer of the code so that it can be stored for later use
        if desired because it can be prohibitively expensive to compute
        :param x: batched tensor of points in ambient dimension
        :return: ManifoldPoint
        """
        return ManifoldPoints(nystrom(x, self.KP)[0])


class SEC():
    """ Class that finds SEC objects for a manifold and can get tangents on that manifold
    """
    def __init__(self, basis_size: int = None):
        self.n1 = basis_size

    def fit(self, manifold: Manifold):
        self.manifold = manifold
        if self.n1 is None:
            self.n1 = min(80, self.manifold.KP.k)
        t2 = time.time()
        ret1 = del1_as_einsum(self.manifold.u, self.manifold.l, torch.diag(self.manifold.peq), self.n1)
        t3 = time.time()
        print(f'Time taken calculating SEC: {t3 - t2:.2f}s')
        self.u1, self.l1, self.d1, _, self.h1, _ = ret1

        return self

    def tangent_basis(self, idxs: List, embedding: str = 'ambient'):
        """ return the tangent basis for all points on the manifold
        @TODO: also implement for ManifoldPoints, not just manifold.KP.X
        @TODO: use arbitrary embedding
        :param idxs: list of which SEC eigenforms you want the vectorfields for
        """
        usize = len(idxs)
        umatrices = torch.permute(self.h1.t() @ self.u1[:, :usize], [1, 0])
        umatrices = umatrices.reshape((usize, self.n1, self.n1)).transpose(1, 2)

        if embedding.lower().startswith('a'):
            vectorfields = torch.tensordot(
                self.manifold.u[:, :self.n1],
                umatrices @ self.manifold.Xhat[:self.n1],
                dims=[[-1], [-2]])
        elif embedding.lower().startswith('u'):
            vectorfields = torch.tensordot(
                self.manifold.u[:, :self.n1], umatrices,
                dims=[[-1], [-2]])
        else:
            raise NotImplementedError

        return vectorfields
