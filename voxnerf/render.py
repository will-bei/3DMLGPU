import numpy as np
import torch
from my3d import unproject

class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2 ** torch.arange(num_freqs).float()  # [num_freqs]

    def forward(self, x):
        """
        x: [N, 3]
        returns: [N, 3 + 3 * 2 * num_freqs]
        """
        out = [x]
        for freq in self.freq_bands.to(x.device):
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)

def subpixel_rays_from_img(H, W, K, c2w_pose, normalize_dir=True, f=8):
    assert c2w_pose[3, 3] == 1.
    H, W = H * f, W * f
    n = H * W
    ys, xs = np.meshgrid(range(H), range(W), indexing="ij")
    xy_coords = np.stack([xs, ys], axis=-1).reshape(n, 2)

    top_left = np.array([-0.5, -0.5]) + 1 / (2 * f)
    xy_coords = top_left + xy_coords / f

    ro = c2w_pose[:, -1]
    pts = unproject(K, xy_coords, depth=1)
    pts = pts @ c2w_pose.T
    rd = pts - ro
    rd = rd[:, :3]
    if normalize_dir:
        rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
    ro = np.tile(ro[:3], (n, 1))
    return ro, rd


def rays_from_img(H, W, K, c2w_pose, normalize_dir=True):
    assert c2w_pose[3, 3] == 1.
    n = H * W
    ys, xs = np.meshgrid(range(H), range(W), indexing="ij")
    xy_coords = np.stack([xs, ys], axis=-1).reshape(n, 2)

    ro = c2w_pose[:, -1]
    pts = unproject(K, xy_coords, depth=1)
    pts = pts @ c2w_pose.T
    rd = pts - ro  # equivalently can subtract [0,0,0,1] before pose transform
    rd = rd[:, :3]
    if normalize_dir:
        rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
    ro = np.tile(ro[:3], (n, 1))
    return ro, rd


def ray_box_intersect(ro, rd, aabb):
    """
    Intersection of ray with axis-aligned bounding box
    This routine works for arbitrary dimensions; commonly d = 2 or 3
    only works for numpy, not torch (which has slightly diff api for min, max, and clone)

    Args:
        ro: [n, d] ray origin
        rd: [n, d] ray direction (assumed to be already normalized;
            if not still fine, meaning of t as time of flight holds true)
        aabb: [d, 2] bbox bound on each dim
    Return:
        is_intersect: [n,] of bool, whether the particular ray intersects the bbox
        t_min: [n,] ray entrance time
        t_max: [n,] ray exit time
    """
    n = ro.shape[0]
    d = aabb.shape[0]
    assert aabb.shape == (d, 2)
    assert ro.shape == (n, d) and rd.shape == (n, d)

    rd = rd.copy()
    rd[rd == 0] = 1e-6  # avoid div overflow; logically safe to give it big t

    ro = ro.reshape(n, d, 1)
    rd = rd.reshape(n, d, 1)
    ts = (aabb - ro) / rd  # [n, d, 2]
    t_min = ts.min(-1).max(-1)  # [n,] last of entrance
    t_max = ts.max(-1).min(-1)  # [n,] first of exit
    is_intersect = t_min < t_max

    return is_intersect, t_min, t_max


def as_torch_tsrs(device, *args):
    ret = []
    for elem in args:
        target_dtype = torch.float32 if np.issubdtype(elem.dtype, np.floating) else None
        ret.append(
            torch.as_tensor(elem, dtype=target_dtype, device=device)
        )
    return ret


def group_mask_filter(mask, *items):
    return [elem[mask] for elem in items]


def mask_back_fill(tsr, N, inds, base_value=1.0):
    shape = [N, *tsr.shape[1:]]
    canvas = base_value * np.ones_like(tsr, shape=shape)
    canvas[inds] = tsr
    return canvas


def render_one_view(model, aabb, H, W, K, pose):
    N = H * W
    bs = max(W * 5, 4096)  # render 5 rows; original batch size 4096, now 4000;

    ro, rd = rays_from_img(H, W, K, pose)
    ro, rd, t_min, t_max, intsct_inds = scene_box_filter(ro, rd, aabb)
    n = len(ro)
    # print(f"{n} vs {N}")  # n can be smaller than N since some rays do not intsct aabb

    # n = n // 1  # actual number of rays to render; only needed for fast debugging

    dev = model.device
    ro, rd, t_min, t_max = as_torch_tsrs(dev, ro, rd, t_min, t_max)
    rgbs = torch.zeros(n, 3, device=dev)
    depth = torch.zeros(n, 1, device=dev)

    with torch.no_grad():
        for i in range(int(np.ceil(n / bs))):
            s = i * bs
            e = min(n, s + bs)
            _rgbs, _depth, _ = render_ray_bundle(
                model, ro[s:e], rd[s:e], t_min[s:e], t_max[s:e]
            )
            rgbs[s:e] = _rgbs
            depth[s:e] = _depth

    rgbs, depth = rgbs.cpu().numpy(), depth.cpu().numpy()

    base_color = 1.0  # empty region needs to be white
    rgbs = mask_back_fill(rgbs, N, intsct_inds, base_color).reshape(H, W, 3)
    depth = mask_back_fill(depth, N, intsct_inds, base_color).reshape(H, W)
    return rgbs, depth


def scene_box_filter(ro, rd, aabb):
    N = len(ro)
    _, t_min, t_max = ray_box_intersect(ro, rd, aabb)
    # do not render what's behind the ray origin
    t_min, t_max = np.maximum(t_min, 0), np.maximum(t_max, 0)
    # can test intersect logic by reducing the focal length
    is_intsct = t_min < t_max
    ro, rd, t_min, t_max = group_mask_filter(is_intsct, ro, rd, t_min, t_max)
    intsct_inds = np.arange(N)[is_intsct]
    return ro, rd, t_min, t_max, intsct_inds

#---------------------------------------------------------------------------------------------------------

def get_gaussian_ray_samples(ro, rd, t_min, step_size, num_steps):
    """
    Returns Gaussian mean and covariance for each sample interval along the ray.
    Assumes uniform sampling from t_min in `step_size` increments.
    Args:
        ro: (n, 3) ray origin
        rd: (n, 3) normalized ray direction
        t_min: (n,) ray entry distances
        step_size: float, step length
        num_steps: int, number of samples
    Returns:
        mean: (k, n, 3)
        cov: (k, n, 3, 3)
    """
    n = ro.shape[0]
    device = ro.device

    t_vals = step_size * torch.arange(num_steps, device=device).view(num_steps, 1)  # (k, 1)
    t0 = t_min.view(1, n) + t_vals            # (k, n)
    t1 = t0 + step_size                       # (k, n)

    # midpoint of the segment: (k, n, 1)
    t_mid = (t0 + t1) / 2.0
    mean = ro[None, :, :] + t_mid[..., None] * rd[None, :, :]  # (k, n, 3)

    delta = (t1 - t0) / 2.0  # (k, n)
    cov_scalar = delta**2 / 3  # for uniform distribution over segment

    # Covariance along direction: rd * rd^T scaled
    d_outer = rd[:, :, None] * rd[:, None, :]  # (n, 3, 3)
    d_outer = d_outer[None, :, :, :]           # (1, n, 3, 3)
    eye = torch.eye(3, device=device).view(1, 1, 3, 3)  # (1, 1, 3, 3)
    cov = cov_scalar[..., None, None] * (eye - d_outer + d_outer * 3)  # (k, n, 3, 3)

    return mean, cov


def integrated_pos_enc(mean, cov, num_freqs=10):
    """
    Integrated positional encoding from Mip-NeRF.
    Args:
        mean: (k, n, 3)
        cov: (k, n, 3, 3)
        num_freqs: number of frequency bands
    Returns:
        encoded: (k * n, 3 * num_freqs * 2)
    """
    k, n, _ = mean.shape
    device = mean.device

    freq_bands = 2 ** torch.arange(num_freqs, device=device).float() * np.pi  # (num_freqs,)
    shape = (1, 1, 1, num_freqs)  # for broadcasting

    mean = mean.view(k, n, 3, 1)  # (k, n, 3, 1)
    cov_diag = torch.diagonal(cov, dim1=-2, dim2=-1)  # (k, n, 3)

    freq = freq_bands.view(1, 1, 1, num_freqs)  # (1, 1, 1, num_freqs)
    scaled_mean = freq * mean  # (k, n, 3, num_freqs)
    scaled_var = (freq ** 2) * cov_diag.view(k, n, 3, 1)  # (k, n, 3, num_freqs)

    # Apply expected value of sin/cos of a Gaussian (Mip-NeRF Eq 7 & 8)
    encoded = torch.cat([
        torch.exp(-0.5 * scaled_var) * torch.sin(scaled_mean),  # (k, n, 3, num_freqs)
        torch.exp(-0.5 * scaled_var) * torch.cos(scaled_mean),
    ], dim=-1)  # (k, n, 3, 2 * num_freqs)

    encoded = encoded.view(k * n, -1)  # flatten (k * n, 3 * 2 * num_freqs)
    return encoded

def render_ray_bundle(model, ro, rd, t_min, t_max):
    """
    model: wrapper with attributes:
      - forward(pts) that returns (density, color)
      - blend_bg_texture, compute_bg(), feats2color() (optional)
      - get_num_samples()
    """
    num_samples, step_size = model.get_num_samples((t_max - t_min).max())
    n, k = len(ro), num_samples

    ticks = step_size * torch.arange(k, device=ro.device)  # [k]
    ticks = ticks.view(k, 1)  # [k, 1]
    t_min = t_min.view(1, n)  # [1, n]
    t_max = t_max.view(n, 1)  # [n, 1] (you can keep this as is if not used in this addition)

    dists = ticks + t_min  # [k, n]

    pts = ro.unsqueeze(0) + rd.unsqueeze(0) * dists.unsqueeze(-1)  # [k, n, 3]
    pts_flat = pts.reshape(-1, 3)  # [k*n, 3]

    # Use model.forward which applies positional encoding and MLP internally
    density, color = model.forward(pts_flat)  # density: [k*n], color: [k*n, 3]

    density = density.view(k, n)
    color = color.view(k, n, 3)

    weights = volume_rend_weights(density, step_size)
    weights3 = weights.unsqueeze(-1)  # [k, n, 1]

    rgbs = (weights3 * color).sum(dim=0)  # [n, 3]

    bg_weight = 1.0 - weights.sum(dim=0)
    bg_weight = bg_weight.unsqueeze(-1)  # [n, 1]

    if getattr(model, 'blend_bg_texture', False):
        uv = spherical_xyz_to_uv(rd)
        bg_feats = model.compute_bg(uv)
        bg_color = model.feats2color(bg_feats)
        rgbs = rgbs + bg_weight * bg_color
    else:
        rgbs = rgbs + bg_weight * 1.0  # white background

    E_dists = (weights * dists).sum(dim=0) + bg_weight.squeeze(-1) * 10.0

    return rgbs, E_dists, weights


def render_ray_bundle__old(model, ro, rd, t_min, t_max):
    num_samples, step_size = model.get_num_samples((t_max - t_min).max())
    n, k = len(ro), num_samples

    # Mip-NeRF style Gaussian ray sample intervals
    mean, cov = get_gaussian_ray_samples(ro, rd, t_min, step_size, k)  # [k, n, 3], [k, n, 3, 3]
    encoded = integrated_pos_enc(mean, cov)  # [k * n, feat_dim]

    # Compute densities and reshape
    σ = model.compute_density_feats(encoded)  # [k * n]
    σ = σ.view(k, n)

    weights = volume_rend_weights(σ, step_size)  # [k, n]
    mask = weights > model.ray_march_weight_thres

    # Compute appearance features and colors
    app_feats = model.compute_app_feats(encoded[mask])  # now from encoded
    c_dim = app_feats.shape[-1]
    colors = torch.zeros(k, n, c_dim, device=ro.device)
    colors[mask] = model.feats2color(app_feats)

    weights = weights.view(k, n, 1)
    bg_weight = 1. - weights.sum(dim=0)  # [n, 1]
    rgbs = (weights * colors).sum(dim=0)  # [n, 3]

    if model.blend_bg_texture:
        uv = spherical_xyz_to_uv(rd)
        bg_feats = model.compute_bg(uv)
        bg_color = model.feats2color(bg_feats)
        rgbs += bg_weight * bg_color
    else:
        rgbs += bg_weight * 1.0  # white bg

    # Expected depth
    ticks = step_size * torch.arange(k, device=ro.device).view(k, 1)
    dists = t_min.view(n, 1) + ticks.T  # [k, n]
    E_dists = (weights.squeeze(-1) * dists).sum(dim=0)
    E_dists += bg_weight.squeeze(-1) * 10.  # constant bg depth

    return rgbs, E_dists, weights.squeeze(-1)



#---------------------------------------------------------------------------------------------------------

def render_ray_bundle_old(model, ro, rd, t_min, t_max):
    """
    The working shape is (k, n, 3) where k is num of samples per ray, n the ray batch size
    During integration the reduction is applied on k

    chain of filtering
    starting with ro, rd (from cameras), and a scene bbox
    - rays that do not intersect scene bbox; sample pts that fall outside the bbox
    - samples that do not fall within alpha mask
    - samples whose densities are very low; no need to compute colors on them
    """
    num_samples, step_size = model.get_num_samples((t_max - t_min).max())
    n, k = len(ro), num_samples

    ticks = step_size * torch.arange(k, device=ro.device)
    ticks = ticks.view(k, 1, 1)
    t_min = t_min.view(n, 1)
    # t_min = t_min + step_size * torch.rand_like(t_min)  # NOTE seems useless
    t_max = t_max.view(n, 1)
    dists = t_min + ticks  # [n, 1], [k, 1, 1] -> [k, n, 1]
    pts = ro + rd * dists  # [n, 3], [n, 3], [k, n, 1] -> [k, n, 3]
    mask = (ticks < (t_max - t_min)).squeeze(-1)  # [k, 1, 1], [n, 1] -> [k, n, 1] -> [k, n]
    smp_pts = pts[mask]

    if model.alphaMask is not None:
        alphas = model.alphaMask.sample_alpha(smp_pts)
        alpha_mask = alphas > 0
        mask[mask.clone()] = alpha_mask
        smp_pts = pts[mask]

    σ = torch.zeros(k, n, device=ro.device)
    σ[mask] = model.compute_density_feats(smp_pts)
    weights = volume_rend_weights(σ, step_size)
    mask = weights > model.ray_march_weight_thres
    smp_pts = pts[mask]

    app_feats = model.compute_app_feats(smp_pts)
    # viewdirs = rd.view(1, n, 3).expand(k, n, 3)[mask]  # ray dirs for each point
    # additional wild factors here as in nerf-w; wild factors are optimizable
    c_dim = app_feats.shape[-1]
    colors = torch.zeros(k, n, c_dim, device=ro.device)
    colors[mask] = model.feats2color(app_feats)

    weights = weights.view(k, n, 1)  # can be used to compute other expected vals e.g. depth
    bg_weight = 1. - weights.sum(dim=0)  # [n, 1]

    rgbs = (weights * colors).sum(dim=0)  # [n, 3]

    if model.blend_bg_texture:
        uv = spherical_xyz_to_uv(rd)
        bg_feats = model.compute_bg(uv)
        bg_color = model.feats2color(bg_feats)
        rgbs = rgbs + bg_weight * bg_color
    else:
        rgbs = rgbs + bg_weight * 1.  # blend white bg color

    # rgbs = rgbs.clamp(0, 1)  # don't clamp since this is can be SD latent features

    E_dists = (weights * dists).sum(dim=0)
    bg_dist = 10.  # blend bg distance; just don't make it too large
    E_dists = E_dists + bg_weight * bg_dist
    return rgbs, E_dists, weights.squeeze(-1)


def spherical_xyz_to_uv(xyz):
    # xyz is Tensor of shape [N, 3], uv in [-1, 1]
    x, y, z = xyz.t()  # [N]
    xy = (x ** 2 + y ** 2) ** 0.5
    u = torch.atan2(xy, z) / torch.pi  # [N]
    v = torch.atan2(y, x) / (torch.pi * 2) + 0.5  # [N]
    uv = torch.stack([u, v], -1)  # [N, 2]
    uv = uv * 2 - 1  # [0, 1] -> [-1, 1]
    return uv


def volume_rend_weights(σ, dist):
    α = 1 - torch.exp(-σ * dist)
    T = torch.ones_like(α)
    T[1:] = (1 - α).cumprod(dim=0)[:-1]
    assert (T >= 0).all()
    weights = α * T
    return weights
