import os
os.environ['FORCE_MEM_EFFICIENT_ATTN'] = "1"
os.environ['XDG_CACHE_HOME'] = 'K:/Weights/'

import gc; gc.enable()

import math
import time
import contextlib
from typing import Callable, Optional, Union, Dict, List

import torch
from einops import rearrange, repeat
from torch import Tensor

import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from .ae import load_ae
from .embedder import HFEmbedder, load_t5, load_clip
from .model import Flux, load_flow_model
from . import t5_path, clip_path, flux_path, offload, is_schnell, ae_path, flux_name, device

class FluxInference:
    
    def __init__(self):
        self.model = None
        self.t5 = None
        self.clip = None
        self.ae = None
        self.device = device
        self.offload = offload
        self.is_schnell = is_schnell
        self.diffusion_step = 0
        
        self.load_models()
    
    def load_models(self):
        if self.model is not None:
            return
        
        self.model = load_flow_model(flux_name, flux_path, device="cpu" if offload else device)
        self.ae = load_ae(flux_name, ae_path, device="cpu" if offload else device)
        self.t5 = load_t5(t5_path, device, max_length=256 if is_schnell else 512)
        self.clip = load_clip(clip_path, device)

    @staticmethod
    def get_noise(
        num_samples: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
    ):
        return torch.randn(
            num_samples,
            16,
            # allow for packing
            2 * math.ceil(height / 16),
            2 * math.ceil(width / 16),
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
    
    @torch.inference_mode()
    def prepare_embedding(self, prompt: str, subprompts: List[str], bs: int = 1):
        if offload:
            torch.cuda.synchronize()
            self.t5 = self.t5.to(device)
            self.clip = self.clip.to(device)

        prompt_emb = self.t5(prompt)
        subprompts_embeds = [self.t5(subprompt) for subprompt in subprompts]
    
        txt = torch.cat([prompt_emb, *subprompts_embeds], dim=1)
        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        txt_ids = torch.zeros(bs, txt.shape[1], 3)

        vec = self.clip(prompt + ', ' + ', '.join(subprompts))
        if vec.shape[0] == 1 and bs > 1:
            vec = repeat(vec, "1 ... -> bs ...", bs=bs)
            
        # offload TEs to CPU, load model to gpu
        if offload:
            self.t5 = self.t5.cpu()
            self.clip = self.clip.cpu()
            torch.cuda.empty_cache()
            
        return txt, txt_ids, vec, subprompts_embeds
    
    def prepare_bboxes(self, bboxes: Dict):
        return None

    def build_attention_mask(self, img: Tensor):
        pass

    def prepare_image(self, img: Tensor, prompt: Union[str, list[str]]):
        bs, c, h, w = img.shape
        if bs == 1 and not isinstance(prompt, str):
            bs = len(prompt)

        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:
            img = repeat(img, "1 ... -> bs ...", bs=bs)

        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        
        return img, img_ids
    
    def encode_init_image(self, init_image, height: int, width: int):

        if isinstance(init_image, np.ndarray):
            init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 255.0
            init_image = init_image.unsqueeze(0) 
        init_image = init_image.to(device)
        init_image = torch.nn.functional.interpolate(init_image, (height, width))
        if offload:
            self.ae.encoder.to(device)
        init_image = self.ae.encode(init_image.to())
        if offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            
        return init_image
    
    def prepare_attention_mask(self, lin_masks: List[Image.Image], reg_embeds: List[Tensor], 
                               Nx: int, emb_size: int, emb_len: int,):
        cross_mask = torch.zeros(emb_len + Nx, emb_len + Nx)
        q_scale = torch.ones(emb_len + Nx)
        k_scale = torch.ones(emb_len + Nx)

        n_regs = len(lin_masks)
        emb_cum_idx = 0

        # Mask main prompt to subprompts
        for j in range(n_regs):
            t1, t2 = emb_cum_idx + (j+1) * emb_size, emb_cum_idx + (j+2) * emb_size
            p1, p2 = emb_cum_idx, emb_cum_idx + emb_size
            print(t1, t2, p1, p2)
            
            cross_mask[t1 : t2, p1 : p2] = 1
            cross_mask[p1 : p2, t1 : t2] = 1
            
        emb_cum_idx += emb_size

        for i, (m, emb) in enumerate(zip(lin_masks, reg_embeds)):
            # mask text
            for j in range(1, n_regs - i):
                t1, t2 = emb_cum_idx + j * emb_size, emb_cum_idx + (j+1) * emb_size
                p1, p2 = emb_cum_idx, emb_cum_idx + emb_size
                print(t1, t2, p1, p2)
                
                cross_mask[t1 : t2, p1 : p2] = 1
                cross_mask[p1 : p2, t1 : t2] = 1
            
            scale = m.sum() / Nx
            print('m: ', m.shape, scale)
            if scale > 1e-5:
                q_scale[emb_cum_idx : emb_cum_idx+emb_size] = 1 / scale
                k_scale[emb_cum_idx : emb_cum_idx+emb_size] = 1 / scale
            
            # m (4096) -> (N_text * 256 + 4096)
            m = torch.cat([torch.ones(emb_size * (n_regs+1)), m])
            print(m.shape)
            
            mb = m > 0.5
            cross_mask[~mb, emb_cum_idx : emb_cum_idx + emb_size] = 1
            cross_mask[emb_cum_idx : emb_cum_idx + emb_size, ~mb] = 1
            emb_cum_idx += emb_size

        # Image Self-Attention attention between different areas blocking
        # Calculate pairwise masks between different areas with the kronecker product
        for i in range(n_regs):
            for j in range(i+1, n_regs):
                # We need to calculate two kr.prod for preserving the symmetry of the matrix
                kron1 = torch.kron(lin_masks[i].unsqueeze(0), lin_masks[j].unsqueeze(-1))
                kron2 = torch.kron(lin_masks[j].unsqueeze(0), lin_masks[i].unsqueeze(-1))
                # cross_mask[emb_cum_idx:, emb_cum_idx:] += kron1 + kron2
                
                # We need to select interesecting regions and set the rows and columns which are intersecting to 0
                
                # Get the intersecting regions
                intersect_idx = torch.logical_and(lin_masks[i] > 0.5, lin_masks[j] > 0.5)
                # Set the intersecting regions to 0
                kron_sum = kron1 + kron2
                kron_sum[intersect_idx, :] = 0
                kron_sum[:, intersect_idx] = 0
                
                # kron_sum[intersect_idx, intersect_idx] = 0

                # Add the kronecker product to the cross mask
                cross_mask[emb_cum_idx:, emb_cum_idx:] += kron_sum
        
        # Clean up the diagonal
        cross_mask.fill_diagonal_(0)
        
        # gt2_mask = cross_mask > 1
        # cross_mask.masked_fill_(gt2_mask, 0)
        
        # Debug kronecker product for case of 2 regions
        # kron1 = torch.kron(lin_masks[0].unsqueeze(0), lin_masks[1].unsqueeze(-1))
        # kron2 = torch.kron(lin_masks[1].unsqueeze(0), lin_masks[0].unsqueeze(-1))

        # cross_mask[emb_cum_idx:, emb_cum_idx:] += kron1 + kron2
            
        q_scale = q_scale.reshape(1, 1, -1, 1).cuda()
        k_scale = k_scale.reshape(1, 1, -1, 1).cuda()
        
        return cross_mask, q_scale, k_scale
    
    @torch.inference_mode()
    def inference_bbox(self, prompt: str, negative_prompt: str,
                       masks: List[Image.Image], subprompts: List[str],
                       aspect_ratio: str, seed: int,
                       guidance: float = 3.5, steps: int = 20,
                       height: int = 1024, width: int = 1024,
                       init_image=None, image2image_strength: float = 0.0):
        t0 = time.perf_counter()
        
        # prompt_emb = self.t5(prompt)
        # reg_embeds = []
        # for p in subprompts:
        #     txt_emb = self.t5(p)
        #     print(f'{txt_emb.shape=}')
        #     reg_embeds.append(txt_emb)
        
        txt_embeds, txt_ids, vec, subprompts_embeds = self.prepare_embedding(prompt, subprompts)

        hH, hW = int(height) // 16, int(width) // 16
        print(height, width, '->', hH, hW)
        lin_masks = []
        for mask in masks:
            mask = mask.convert('L')
            mask = torch.tensor(np.array(mask)).unsqueeze(0).unsqueeze(0) / 255
            mask = torch.nn.functional.interpolate(mask, (hH, hW), mode='nearest-exact').flatten()
            # Linearize mask
            lin_masks.append(mask)

        emb_size = 256
        Nx = int(hH * hW)
        emb_len = (len(subprompts_embeds) + 1) * 256
        
        cross_mask, q_scale, k_scale = self.prepare_attention_mask(
            lin_masks, subprompts_embeds, Nx, emb_size, emb_len)
        
        # Visualize and save the attention mask
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(cross_mask.cpu().numpy())
        plt.savefig('attention_mask.png')
        
        if init_image is not None:
            init_image = self.encode_init_image(init_image)
            
        # prepare input
        x = self.get_noise(
            1,
            height,
            width,
            device=device,
            dtype=torch.bfloat16,
            seed=seed,
        )
        img, img_ids = self.prepare_image(x, prompt)
        # Move to device
        img, img_ids = img.to(device), img_ids.to(device)
        vec, txt_embeds, txt_ids = vec.to(device), txt_embeds.to(device), txt_ids.to(device)
        
        timesteps = get_schedule(
            steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=(not is_schnell),
        )
        if init_image is not None:
            t_idx = int((1 - image2image_strength) * steps)
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image.to(x.dtype)
        
        if offload:
            self.model = self.model.to(device)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # denoise initial noise
            diffusion_step = 0
            attn_bias = cross_mask.to(device=img.device, dtype=img.dtype)
            attn_mask_bool = attn_bias > 0.5
            attn_bias.masked_fill_(attn_mask_bool, float('-inf'))
            
            # this is ignored for schnell
            guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
            for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:])):
                t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
                pred = self.model(
                    img=img,
                    img_ids=img_ids,
                    txt=txt_embeds,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    attn_kwargs={'attn_mask': attn_bias, 'q_scale': q_scale, 'k_scale': k_scale}
                )

                img = img + (t_prev - t_curr) * pred
                diffusion_step += 1
                
                del pred
                
            x = img
            del attn_bias, attn_mask_bool, guidance_vec

        if offload:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            gc.collect()
        
        img = self.decode_latents(x, height, width)

        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s.")
        
        return img
    
    @torch.inference_mode()
    def decode_latents(self, x: Tensor, height: int, width: int):
        @contextlib.contextmanager
        def ae_on_device():
            if self.offload:
                self.ae = self.ae.to(x.device)
            try:
                yield
            finally:
                if self.offload:
                    self.ae = self.ae.cpu()
                    torch.cuda.empty_cache()
                    gc.collect()
            
        with ae_on_device():
            # decode latents to pixel space
            x = unpack(x.float(), height, width)
            print(f'unpacked x: {x.shape}')
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                x = self.ae.decode(x)
                print(f'decoded x: {x.shape}')
        
            # bring into PIL format
            x = x.clamp(-1, 1).float()
            x = rearrange(x[0], "c h w -> h w c")

        img_arr = (127.5 * (x.cpu() + 1.0)).clamp(0, 255).byte().numpy()
        del x
        
        return img_arr


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
