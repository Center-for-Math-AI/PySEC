import os, sys, time
import numpy as np
import torch
from PySEC.distance_funcs import *

devices = ['cuda:0', 'cpu']
batch_sizes = [128, 512, 2048]
times = torch.empty((len(devices), len(batch_sizes)))

x = torch.rand((80+10**3, 1, 40, 40))
y = torch.rand((80+10**3, 1, 40, 40))
ret0 = None

for iid, device in enumerate(devices):

    x, y = x.to(device), y.to(device)

    for iib, batch_size in enumerate(batch_sizes):
        t0 = time.time()
        ret1 = pdist2(x, distance='ssim', compute_device=device, batch_size=batch_size, progress=True)
        t1 = time.time()

        times[iid, iib] = t1 - t0
        if ret0 is not None:
            assert torch.allclose(ret0.to('cpu'), ret1.to('cpu'))
        ret0 = ret1


fstr = "{:10} ::" + " {:6.0f} |" * len(batch_sizes)
print(fstr.format('dev/Batch', *batch_sizes))
fstr = "{:10} ::" + " {:6.2f} |" * len(batch_sizes)
for iid, device in enumerate(devices):
    print(fstr.format(device, *times[iid].tolist()))


# t0 = time.time()
# ret1 = pdist2(x, x, distance='ssim')
# t1 = time.time()
# sys.stderr.write('\n'), sys.stderr.flush()
# print(), sys.stdout.flush()
# print(f"pdist2 time for {x.shape[0]}x{y.shape[0]} :: {t1-t0:.2f}s")
# print(), sys.stdout.flush()
#
# t0 = time.time()
# ret2 = pdist2(x, distance='ssim')
# t1 = time.time()
# sys.stderr.write('\n'), sys.stderr.flush()
# print(), sys.stdout.flush()
# print(f"pdist2 time for {x.shape[0]}x{y.shape[0]} :: {t1-t0:.2f}s")
# print(), sys.stdout.flush()
#
# print(f"total diff norm: {torch.linalg.norm(ret1-ret2).item():.4e}")
# idx_max = torch.argmax(torch.abs(ret1-ret2))
# idx2 = np.unravel_index(idx_max.item(), ret1.shape)
# print(f"Most different elements: ")
# print(f"ret1[{idx2[0]},{idx2[1]}]: {ret1[idx2]:.8e}")
# print(f"ret2[{idx2[0]},{idx2[1]}]: {ret2[idx2]:.8e}")
# print(f"diff: {ret1[idx2]-ret2[idx2]:4e}")

debug_var = 1


