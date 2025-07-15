import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from core.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)

        import importlib
        mod = importlib.import_module(f'models.ours.{module_name}')
        if hasattr(mod, 'UNetWithIPAdapter'):
            UNet = getattr(mod, 'UNetWithIPAdapter')
        else:
            UNet = getattr(mod, 'UNet')

        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        self.betas = betas
        alphas = 1. - betas
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.denoise_fn(y_t, noise_level, y_cond)
        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)
        model_mean, posterior_log_variance = self.q_posterior(y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
        b, *_ = y_cond.shape
        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps // sample_num)
        y_t = default(y_t, lambda: torch.randn_like(y_0))
        ret_arr = y_t
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.from_numpy(self.betas))
        model_fn = model_wrapper(
            self.denoise_fn,
            noise_schedule,
            model_type="x_start",
            model_kwargs={"cond_img": y_cond},
            guidance_type="classifier-free",
            condition=None,
            unconditional_condition=None,
            guidance_scale=1.,
        )
        dpm_solver = DPM_Solver(
            model_fn,
            noise_schedule,
            algorithm_type="dpmsolver++",
            correcting_x0_fn="dynamic_thresholding",
        )
        y_t = dpm_solver.sample(
            y_t,
            steps=20,
            order=2,
            skip_type="time_uniform",
            method="multistep",
            denoise_to_zero=True,
        )
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)
        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)
        if mask is not None:
            noise_hat = self.denoise_fn(y_noisy * mask + (1. - mask) * y_0, sample_gammas, y_cond)
            loss = self.loss_fn(mask * noise, mask * noise_hat)
        else:
            y_0_hat = self.denoise_fn(y_noisy, sample_gammas, y_cond)
            loss = self.loss_fn(y_0, y_0_hat)
        return loss


# utility functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clip(betas, 0, 0.999)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timestep)
        betas = torch.sigmoid(betas) * (linear_end - linear_start) + linear_start
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return betas
