import os, sys
import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from imageio import imwrite
from pydantic import validator

from my.utils import (
    tqdm, EventStorage, HeartBeat, EarlyLoopBreak,
    get_event_storage, get_heartbeat, read_stats
)
from my.config import BaseConf, dispatch, optional_load_config
from my.utils.seed import seed_everything

from adapt import ScoreAdapter, karras_t_schedule
from run_img_sampling import GDDPM, SD, StableDiffusion
from misc import torch_samps_to_imgs
from pose import PoseConfig

from run_nerf import VoxConfig
from voxnerf.utils import every
from voxnerf.render import (
    as_torch_tsrs, rays_from_img, ray_box_intersect, render_ray_bundle
)
from voxnerf.vis import stitch_vis, bad_vis as nerf_vis

from torch.profiler import profile, record_function, ProfilerActivity

device_glb = torch.device("cuda")

import json
from torch.profiler import profile, ProfilerActivity

class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2 ** torch.arange(num_freqs) * np.pi

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


class SimpleNeRFMLP(torch.nn.Module):
    def __init__(self, input_dim=3 + 3 * 2 * 10, hidden_dim=128):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 4)  # sigma + rgb (r,g,b)
        )

    def forward(self, x):
        out = self.layers(x)
        sigma = torch.relu(out[:, 0])
        rgb = torch.sigmoid(out[:, 1:4])
        return sigma, rgb

class MLPScoreAdapter:
    def __init__(self, mlp, prompt, pos_encoder=None):
        self.mlp = mlp
        self.pos_encoder = pos_encoder
        self.device = next(mlp.parameters()).device
        self.us = torch.linspace(0.01, 1.0, steps=100).to(self.device)
        self.prompt = prompt

    def samps_centered(self):
        # Return True or False depending on your model's sampling convention
        return True

    def data_shape(self):
        # Return the shape of the data your model expects, e.g. (channels, height, width)
        return (4, 64, 64)  # Example shape

    def forward(self, pts):
        if self.pos_encoder is not None:
            # assume pts is (B, 3, H, W) or (N, 3)
            if pts.dim() == 4:
                pts = pts.permute(0, 2, 3, 1).reshape(-1, 3)  # flatten spatial dims
            pts_enc = self.pos_encoder(pts)
        else:
            pts_enc = pts
            if pts_enc.dim() > 2:
                pts_enc = pts_enc.view(-1, pts_enc.shape[-1])

        print("[DEBUG] pts_enc shape:", pts_enc.shape)
        print("[DEBUG] MLP input weight shape:", self.mlp.layers[0].weight.shape)

        sigma, rgb = self.mlp(pts_enc)
        return sigma, rgb

    def denoise(self, zs, sigma, **score_conds):
        # Implement denoising logic based on zs, sigma, and any conditioning
        # If no special denoising, just forward through the network
        return self.forward(zs)[1]  # return rgb part as example

    def prompts_emb(self, prompts):
        # If your model supports prompt embeddings, implement this
        # Otherwise return empty dict or None
        return {}
    
    def get_num_samples(self, max_dist):
        num_samples = 64
        step_size = max_dist / num_samples
        return num_samples, step_size
    
    def opt_params(self):
        return self.mlp.parameters()

TRAIN_WAIT_STEPS = 1
TRAIN_WARMUP_STEPS = 2
TRAIN_ACTIVE_STEPS = 5
TRAIN_REPEAT_STEPS = 1
train_prof_schedule = torch.profiler.schedule(
wait=TRAIN_WAIT_STEPS,
warmup=TRAIN_WARMUP_STEPS,
active=TRAIN_ACTIVE_STEPS,
repeat=TRAIN_REPEAT_STEPS)    
prof = torch.profiler.profile(
    schedule=train_prof_schedule,
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    on_trace_ready=lambda prof: (
        print(f"Saving trace to {os.path.join(os.getcwd(), 'trace.json')}"),
        prof.export_chrome_trace(os.path.join(os.getcwd(), 'trace.json'))
    ),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
)


def tsr_stats(tsr):
    return {
        "mean": tsr.mean().item(),
        "std": tsr.std().item(),
        "max": tsr.max().item(),
    }


class SJC(BaseConf):
    family:     str = "sd"
    gddpm:      GDDPM = GDDPM()
    sd:         SD = SD(
        variant="v1",
        prompt="A high quality photo of a delicious burger",
        scale=100.0
    )
    lr:         float = 0.05
    n_steps:    int = 1000
    vox:        VoxConfig = VoxConfig(
        model_type="V_SD", grid_size=100, density_shift=-1.0, c=3,
        blend_bg_texture=True, bg_texture_hw=4,
        bbox_len=1.0
    )
    pose:       PoseConfig = PoseConfig(rend_hw=64, FoV=60.0, R=1.5)

    emptiness_scale:    int = 10
    emptiness_weight:   int = 1e4
    emptiness_step:     float = 0.5
    emptiness_multiplier: float = 20.0

    depth_weight:       int = 0

    var_red:     bool = True

    @validator("vox")
    def check_vox(cls, vox_cfg, values):
        family = values['family']
        if family == "sd":
            vox_cfg.c = 4
        return vox_cfg

    def run(self):
        cfgs = self.dict()

        family = cfgs.pop("family")

        # Instantiate raw MLP and positional encoder
        device = device_glb
        pos_encoder = PositionalEncoding(num_freqs=10).to(device)
        mlp = SimpleNeRFMLP(input_dim=3 + 3 * 2 * 10).to(device)
        print(f"[DEBUG] MLP input layer weight shape: {mlp.layers[0].weight.shape}")
        prompt = self.sd.prompt
        model = MLPScoreAdapter(mlp, prompt, pos_encoder)

        cfgs.pop("vox")
        vox = self.vox.make()

        cfgs.pop("pose")
        poser = self.pose.make()

        sjc_3d(**cfgs, poser=poser, model=model, vox=vox)


def sjc_3d(
    poser, vox, model: ScoreAdapter,
    lr, n_steps, emptiness_scale, emptiness_weight, emptiness_step, emptiness_multiplier,
    depth_weight, var_red, **kwargs
):
    del kwargs

    assert model.samps_centered()
    _, target_H, target_W = model.data_shape()
    bs = 1
    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)
    opt = torch.optim.Adamax(vox.opt_params(), lr=lr)

    H, W = poser.H, poser.W
    Ks, poses, prompt_prefixes = poser.sample_train(n_steps)

    ts = model.us[30:-10]
    fuse = EarlyLoopBreak(5)

    same_noise = torch.randn(1, 4, H, W, device=model.device).repeat(bs, 1, 1, 1)

    prof.start() 
    with tqdm(total=n_steps) as pbar, \
        HeartBeat(pbar) as hbeat, \
            EventStorage() as metric:
        for i in range(n_steps):
            if fuse.on_break():
                break

            p = f"{prompt_prefixes[i]} {model.prompt}"
            score_conds = model.prompts_emb([p])

            with record_function("forward_pass"):
                y, depth, ws = render_one_view(vox, aabb, H, W, Ks[i], poses[i], model, return_w=True)
            if isinstance(model, StableDiffusion):
                pass
            else:
                y = torch.nn.functional.interpolate(y, (target_H, target_W), mode='bilinear')

            opt.zero_grad()
            
            with record_function("loss_calc"):
                with torch.no_grad():
                    ts_np = ts.cpu().numpy() if isinstance(ts, torch.Tensor) else ts
                    chosen_σs = np.random.choice(ts_np, bs, replace=False)
                    chosen_σs = chosen_σs.reshape(-1, 1, 1, 1)
                    chosen_σs = torch.as_tensor(chosen_σs, device=model.device, dtype=torch.float32)

                    # chosen_σs = us[i]

                    noise = torch.randn(bs, *y.shape[1:], device=model.device)

                    zs = y + chosen_σs * noise
                    Ds = model.denoise(zs, chosen_σs, **score_conds)

                    # Extract RGB part if needed
                    if Ds.shape[1] > 3 and y.shape[1] == 3:
                        Dsrgb = Ds[:, :3, :, :]
                    else:
                        Dsrgb = Ds

                    # Resize Dsrgb to match y's spatial size
                    if Dsrgb.shape[2:] != y.shape[2:]:
                        Dsrgb = torch.nn.functional.interpolate(Dsrgb, size=y.shape[2:], mode='bilinear', align_corners=False)

                    if var_red:
                        grad = (Dsrgb - y) / chosen_σs
                    else:
                        grad = (Dsrgb - zs) / chosen_σs

                    grad = grad.mean(0, keepdim=True)
            
            with record_function("backward_pass"):
                y.backward(-grad, retain_graph=True)

            if depth_weight > 0:
                center_depth = depth[7:-7, 7:-7]
                border_depth_mean = (depth.sum() - center_depth.sum()) / (64*64-50*50)
                center_depth_mean = center_depth.mean()
                depth_diff = center_depth_mean - border_depth_mean
                depth_loss = - torch.log(depth_diff + 1e-12)
                depth_loss = depth_weight * depth_loss
                depth_loss.backward(retain_graph=True)

            emptiness_loss = torch.log(1 + emptiness_scale * ws).mean()
            emptiness_loss = emptiness_weight * emptiness_loss
            if emptiness_step * n_steps <= i:
                emptiness_loss *= emptiness_multiplier
            emptiness_loss.backward()

            with record_function("optimization"):
                opt.step()

            prof.step() 

            metric.put_scalars(**tsr_stats(y))

            if every(pbar, percent=1):
                with torch.no_grad():
                    if isinstance(model, StableDiffusion):
                        y = model.decode(y)
                    vis_routine(metric, y, depth)

            # if every(pbar, step=2500):
            #     metric.put_artifact(
            #         "ckpt", ".pt", lambda fn: torch.save(vox.state_dict(), fn)
            #     )
            #     with EventStorage("test"):
            #         evaluate(model, vox, poser)

            metric.step()
            pbar.update()
            pbar.set_description(p)
            hbeat.beat()

        metric.put_artifact(
            "ckpt", ".pt", lambda fn: torch.save(vox.state_dict(), fn)
        )
        with EventStorage("test"):
            evaluate(model, vox, poser)

        metric.step()

        hbeat.done()

    prof.stop()
    print("\nTop 30 results sorted by CPU time avg:\n")
    print(prof.key_averages().table(sort_by="cpu_time_str", row_limit=30))

    print("\nTop 30 results sorted by CUDA time avg:\n")
    print(prof.key_averages().table(sort_by="cuda_time_str", row_limit=30))

    # Get the profiler output as a formatted string
    prof_table_str = prof.key_averages().table(sort_by="cpu_time_str", row_limit=30)

    # Save to a text file
    with open("torch_profiling.txt", "w") as text_file:
        text_file.write(prof_table_str)

    print("Text file saved successfully as 'torch_profiling.txt'!")

@torch.no_grad()
def evaluate(score_model, vox, poser):
    H, W = poser.H, poser.W
    vox.eval()
    K, poses = poser.sample_test(100)

    fuse = EarlyLoopBreak(5)
    metric = get_event_storage()
    hbeat = get_heartbeat()

    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)

    num_imgs = len(poses)

    for i in (pbar := tqdm(range(num_imgs))):
        if fuse.on_break():
            break

        pose = poses[i]
        y, depth = render_one_view(vox, aabb, H, W, K, pose, model)
        if isinstance(score_model, StableDiffusion):
            y = score_model.decode(y)
        vis_routine(metric, y, depth)

        metric.step()
        hbeat.beat()

    metric.flush_history()

    metric.put_artifact(
        "view_seq", ".mp4",
        lambda fn: stitch_vis(fn, read_stats(metric.output_dir, "view")[1])
    )

    metric.step()


def render_one_view(vox, aabb, H, W, K, pose, model, return_w=False):
    N = H * W
    ro, rd = rays_from_img(H, W, K, pose)
    ro, rd, t_min, t_max = scene_box_filter(ro, rd, aabb)
    assert len(ro) == N, "for now all pixels must be in"
    ro, rd, t_min, t_max = as_torch_tsrs(vox.device, ro, rd, t_min, t_max)
    rgbs, depth, weights = render_ray_bundle(model, ro, rd, t_min, t_max)

    rgbs = rearrange(rgbs, "(h w) c -> 1 c h w", h=H, w=W)
    depth = rearrange(depth.unsqueeze(-1), "(h w) 1 -> h w", h=H, w=W)
    if return_w:
        return rgbs, depth, weights
    else:
        return rgbs, depth


def scene_box_filter(ro, rd, aabb):
    _, t_min, t_max = ray_box_intersect(ro, rd, aabb)
    # do not render what's behind the ray origin
    t_min, t_max = np.maximum(t_min, 0), np.maximum(t_max, 0)
    return ro, rd, t_min, t_max


def vis_routine(metric, y, depth):
    pane = nerf_vis(y, depth, final_H=256)
    im = torch_samps_to_imgs(y)[0]
    depth = depth.cpu().numpy()
    metric.put_artifact("view", ".png", lambda fn: imwrite(fn, pane))
    metric.put_artifact("img", ".png", lambda fn: imwrite(fn, im))
    metric.put_artifact("depth", ".npy", lambda fn: np.save(fn, depth))


def evaluate_ckpt():
    cfg = optional_load_config(fname="full_config.yml")
    assert len(cfg) > 0, "can't find cfg file"
    mod = SJC(**cfg)

    family = cfg.pop("family")
    model: ScoreAdapter = getattr(mod, family).make()
    vox = mod.vox.make()
    poser = mod.pose.make()

    pbar = tqdm(range(1))

    with EventStorage(), HeartBeat(pbar):
        ckpt_fname = latest_ckpt()
        state = torch.load(ckpt_fname, map_location="cpu")
        vox.load_state_dict(state)
        vox.to(device_glb)

        with EventStorage("test"):
            evaluate(model, vox, poser)


def latest_ckpt():
    ts, ys = read_stats("./", "ckpt")
    assert len(ys) > 0
    return ys[-1]


if __name__ == "__main__":
    seed_everything(0)
    dispatch(SJC)
    # evaluate_ckpt()
