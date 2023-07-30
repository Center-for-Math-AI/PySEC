import numpy as np
import matplotlib.pyplot as plt

x = np.logspace(0, 7, num=20, endpoint=True)

ks = np.array([10, 20, 50])
ds = np.array([128**2, 256**2, 1080**2])
ks = np.array([10, 20])
ds = np.array([128**2, 256**2])
ds = np.array([40**2, 256**2,])

def comp1(n, k, d=1):
    return k * n + n * d

def comp2(n, k, d=1):
    return n * k * d

fig, ax = plt.subplots(1, 1)

for ik, k in enumerate(ks):
    for id, d in enumerate(ds):
        y1 = comp1(x, k, d)
        label1 = f'y1(x, {k}, {int(np.sqrt(d))}^2)'
        ax.plot(x, y1, label=label1, zorder=40, linestyle=':')

        y2 = comp2(x, k, d)
        label2 = f'y2(x, {k}, {int(np.sqrt(d))}^2)'
        ax.plot(x, y2, label=label2, zorder=20)

ax.set_xscale("log", base=10); ax.set_yscale("log", base=10)
ax.legend()
plt.show(), plt.close(fig)

debug_var = 1

