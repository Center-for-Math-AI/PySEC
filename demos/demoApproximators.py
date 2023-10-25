import os, time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PySEC import generate_data as gd
from PySEC.approximators import Manifold, SEC

web_plot = False
if web_plot:
    import matplotlib
    matplotlib.use('WebAgg')

torch.backends.cudnn.deterministic = True  #
torch.manual_seed(0)

# get 3d data points of a klein bottle projected in 3d
u, v = np.linspace(0, 2*np.pi, 40), np.linspace(0, 2*np.pi, 40)
ux, vx = np.meshgrid(u,v)
x, y, z = gd.klein_bottle_3d(torch.as_tensor(ux), torch.as_tensor(vx))
print(f'Size of 3d mesh: {3 * x.nelement() * x.element_size() / 2**20:.2f} MB')

t0 = time.time()
num_images = 181
# generate images of the klein bottle from different views
images = np.empty((num_images, 100, 100, 3), dtype=np.uint8)
intrinsic_params = np.empty((num_images, 1), dtype=float)
for ii, az in enumerate(np.linspace(0., 360., num_images, endpoint=False)):
    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.axis('off')
    ax.view_init(az, az)
    ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap = plt.get_cmap('inferno'),
                           linewidth = 0, antialiased = False)
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    images[ii] = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy()
    intrinsic_params[ii] = float(az)
    plt.close(fig)

t1 = time.time()
shape_str = 'x'.join([str(s) for s in images[0].shape])
print(f'Time to generate {len(images)} images with shape {shape_str}: {t1-t0:.2f}s')


# look at some of the images
fig, aax = plt.subplots(3, 3, figsize=(8, 8))
for ii, ax in enumerate(aax.flat):
    idx = ii * round(len(images) / len(aax.flat))
    ax.imshow(images[idx])
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f'View angle: {intrinsic_params[idx, 0]:.1f}')

fig.suptitle('Samples of dataset images for building a manifold')
fig.tight_layout()
# plt.show(), plt.close(fig)
fig.show()

data = torch.as_tensor(images, dtype=torch.float)
ip = torch.as_tensor(intrinsic_params, dtype=torch.float32)
print(f'Size of generated image dataset: {data.nelement() * data.element_size() / 2**20:.2f}MB')

# build a manifold approximator
geom_man = Manifold().fit(data, ip)


# now look at the manifold if some different combos of diffusion map coords
gip = geom_man.intrinsic_params
plot_vec_list = [
    [1, 2, 3],
    # [1, 2, 4],
    [1, 2, 5],
    [3, 4, 1],
    [2, 3, 5],
]
view_az, view_dl = 20, 40
fig = plt.figure(figsize=(8, 8))
for ii, plot_vecs in enumerate(plot_vec_list):
    ax = fig.add_subplot(2, 2, 1 + ii, projection='3d')
    ax.view_init(view_dl, view_az), ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
    ax.set_xlabel(f'U{plot_vecs[0]}', labelpad=-6), ax.set_ylabel(f'U{plot_vecs[1]}', labelpad=-6), ax.set_zlabel(f'U{plot_vecs[2]}', labelpad=-6)
    sax = ax.scatter(*geom_man.u[:, plot_vecs].t().cpu(), c=gip[:], cmap='hsv', s=3)
    ax.set_title(f'coords: {plot_vecs}')

fig.suptitle('Manifold in diffusion coordinates, colored by view angle')
fig.tight_layout()
# plt.show(), plt.close(fig)
fig.show()


# now noise some dataset images and map them back to the manifold
noise_level = 0.2
idxs = [20, 40, 63, 120]
fig, aax = plt.subplots(3, len(idxs), figsize=(len(idxs)*2, 6))
for ii, idx in enumerate(idxs):
    noisy_data = noise_level * 255. * (torch.rand_like(data[idx]) - 0.5)
    noisy_image = torch.clamp(data[idx] + noisy_data, min=0., max=255.)
    recon_image, recon_params = geom_man.embed(noisy_image.unsqueeze(0), embedding=['ambient', 'intrinsic'])
    recon_image = recon_image.clamp(0., 255.)

    ax = aax[0, ii]
    ax.imshow(data[idx].to(torch.uint8).numpy())
    ax.axis('off')
    ax.set_title(f'Image @ {geom_man.intrinsic_params[idx, 0]:.1f}')
    ax = aax[1, ii]
    ax.imshow(noisy_image.to(torch.uint8).numpy())
    ax.axis('off')
    ax.set_title(f'Noisy image')
    ax = aax[2, ii]
    ax.imshow(recon_image[0].to(torch.uint8).numpy())
    ax.axis('off')
    ax.set_title(f'Recon @ {recon_params[0, 0]:.2f}')

fig.suptitle(f'Noisy image reconstruction on manifold @ {noise_level:.2f} std dev noise')
fig.tight_layout()
# plt.show()
fig.show()


# now compute SEC for the manifold
geom_sec = SEC().fit(geom_man)

# now plot some SEC vector fields
vectorfields = geom_sec.tangent_basis(list(range(10)), embedding='ubasis')
tan_ss = slice(None, None, 4) # subset of points for plot to be less busy
plot_vecs = plot_vec_list[0]
fig = plt.figure(figsize=(8, 8))
for uvec in range(4):
    vectorfield = vectorfields[:, uvec]
    ax = fig.add_subplot(2, 2, 1 + uvec, projection='3d')
    sax = ax.scatter(*geom_man.u[tan_ss, plot_vecs].t().cpu(), c='gray', alpha=0.6, s=5)
    ax.view_init(view_dl, view_az), ax.set_xticklabels([]), ax.set_yticklabels([]), ax.set_zticklabels([])
    tan_len_fac = 1.e0 / torch.linalg.norm(vectorfields[:, 0]).item()
    tax = ax.quiver(*geom_man.u[tan_ss, plot_vecs].t().cpu(), *vectorfield[tan_ss, plot_vecs].t().cpu(),
                    length = 2 * tan_len_fac, normalize = False, color = 'red', arrow_length_ratio = 0.4, alpha = 0.6, )
    # ax.set_title(r'|$\Delta_1$' + f' Tan({uvec})|: {torch.linalg.norm(vectorfield):.2e}')
    ax.set_title(r'|$\Delta_1$' + f' Tan({uvec})|: {geom_sec.l1[uvec]:.2f}')
    ax.set_xlabel(f'U{plot_vecs[0]}', labelpad=-6), ax.set_ylabel(f'U{plot_vecs[1]}', labelpad=-6), ax.set_zlabel(
        f'U{plot_vecs[2]}', labelpad=-6)

fig.suptitle('SEC vectorfields')
fig.tight_layout()
# plt.show(), plt.close(fig)
fig.show()

debug_var = 1