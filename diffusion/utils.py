import os
import re
import sys
import math
import torch
import numpy as np
from PIL import Image
from itertools import repeat
from collections.abc import Iterable
import logging
import torch.distributed as dist
from mmcv.utils.logging import logger_initialized

import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


def resize_and_crop_img(img: Image, new_width, new_height):
    orig_width, orig_height = img.size

    ratio = max(new_width/orig_width, new_height/orig_height)
    resized_width = int(orig_width * ratio)
    resized_height = int(orig_height * ratio)

    img = img.resize((resized_width, resized_height), Image.LANCZOS)

    left = (resized_width - new_width)/2
    top = (resized_height - new_height)/2
    right = (resized_width + new_width)/2
    bottom = (resized_height + new_height)/2

    img = img.crop((left, top, right, bottom))

    return img


def mask_feature(emb, mask):
    if emb.shape[0] == 1:
        keep_index = mask.sum().item()
        return emb[:, :, :keep_index, :], keep_index
    else:
        masked_feature = emb * mask[:, None, :, None]
        return masked_feature, emb.shape[2]


def prepare_prompt_ar(prompt, ratios, device='cpu', show=True):
    # get aspect_ratio or ar
    aspect_ratios = re.findall(r"--aspect_ratio\s+(\d+:\d+)", prompt)
    ars = re.findall(r"--ar\s+(\d+:\d+)", prompt)
    custom_hw = re.findall(r"--hw\s+(\d+:\d+)", prompt)
    if show:
        print("aspect_ratios:", aspect_ratios, "ars:", ars, "hws:", custom_hw)
    prompt_clean = prompt.split("--aspect_ratio")[0].split("--ar")[0].split("--hw")[0]
    if len(aspect_ratios) + len(ars) + len(custom_hw) == 0 and show:
        print("Wrong prompt format. Set to default ar: 1. change your prompt into format '--ar h:w or --hw h:w' for correct generating")
    if len(aspect_ratios) != 0:
        ar = float(aspect_ratios[0].split(':')[0]) / float(aspect_ratios[0].split(':')[1])
    elif len(ars) != 0:
        ar = float(ars[0].split(':')[0]) / float(ars[0].split(':')[1])
    else:
        ar = 1.
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    if len(custom_hw) != 0:
        custom_hw = [float(custom_hw[0].split(':')[0]), float(custom_hw[0].split(':')[1])]
    else:
        custom_hw = ratios[closest_ratio]
    default_hw = ratios[closest_ratio]
    prompt_show = f'prompt: {prompt_clean.strip()}\nSize: --ar {closest_ratio}, --bin hw {ratios[closest_ratio]}, --custom hw {custom_hw}'
    return prompt_clean, prompt_show, torch.tensor(default_hw, device=device)[None], torch.tensor([float(closest_ratio)], device=device)[None], torch.tensor(custom_hw, device=device)[None]


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    return local_rank


def is_master():
    return get_rank() == 0


def is_local_master():
    return get_local_rank() == 0


def get_root_logger(log_file=None, log_level=logging.INFO, name='PixArt'):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str): logger name
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    if log_file is None:
        log_file = '/dev/null'
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)
    return logger


def get_logger(name, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    logger.propagate = False  # disable root logger to avoid duplicate logging

    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    # only rank0 for each node will print logs
    log_level = log_level if is_local_master() else logging.ERROR
    logger.setLevel(log_level)

    logger_initialized[name] = True

    return logger


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def set_grad_checkpoint(model, use_fp32_attention=False, gc_step=1):
    assert isinstance(model, nn.Module)

    def set_attr(module):
        module.grad_checkpointing = True
        module.fp32_attention = use_fp32_attention
        module.grad_checkpointing_step = gc_step
    model.apply(set_attr)


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, 'grad_checkpointing', False):
        if isinstance(module, Iterable):
            gc_step = module[0].grad_checkpointing_step
            return checkpoint_sequential(module, gc_step, *args, **kwargs)
        else:
            return checkpoint(module, *args, **kwargs)
    return module(*args, **kwargs)


ASPECT_RATIO_2880 = {
    '0.25': [1408.0, 5760.0], '0.26': [1408.0, 5568.0], '0.27': [1408.0, 5376.0], '0.28': [1408.0, 5184.0],
    '0.32': [1600.0, 4992.0], '0.33': [1600.0, 4800.0], '0.34': [1600.0, 4672.0], '0.4': [1792.0, 4480.0],
    '0.42': [1792.0, 4288.0], '0.47': [1920.0, 4096.0], '0.49': [1920.0, 3904.0], '0.51': [1920.0, 3776.0],
    '0.55': [2112.0, 3840.0], '0.59': [2112.0, 3584.0], '0.68': [2304.0, 3392.0], '0.72': [2304.0, 3200.0],
    '0.78': [2496.0, 3200.0], '0.83': [2496.0, 3008.0], '0.89': [2688.0, 3008.0], '0.93': [2688.0, 2880.0],
    '1.0': [2880.0, 2880.0], '1.07': [2880.0, 2688.0], '1.12': [3008.0, 2688.0], '1.21': [3008.0, 2496.0],
    '1.28': [3200.0, 2496.0], '1.39': [3200.0, 2304.0], '1.47': [3392.0, 2304.0], '1.7': [3584.0, 2112.0],
    '1.82': [3840.0, 2112.0], '2.03': [3904.0, 1920.0], '2.13': [4096.0, 1920.0], '2.39': [4288.0, 1792.0],
    '2.5': [4480.0, 1792.0], '2.92': [4672.0, 1600.0], '3.0': [4800.0, 1600.0], '3.12': [4992.0, 1600.0],
    '3.68': [5184.0, 1408.0], '3.82': [5376.0, 1408.0], '3.95': [5568.0, 1408.0], '4.0': [5760.0, 1408.0]
}

ASPECT_RATIO_2048 = {
    '0.25': [1024.0, 4096.0], '0.26': [1024.0, 3968.0], '0.27': [1024.0, 3840.0], '0.28': [1024.0, 3712.0],
    '0.32': [1152.0, 3584.0], '0.33': [1152.0, 3456.0], '0.35': [1152.0, 3328.0], '0.4': [1280.0, 3200.0],
    '0.42': [1280.0, 3072.0], '0.48': [1408.0, 2944.0], '0.5': [1408.0, 2816.0], '0.52': [1408.0, 2688.0],
    '0.57': [1536.0, 2688.0], '0.6': [1536.0, 2560.0], '0.68': [1664.0, 2432.0], '0.72': [1664.0, 2304.0],
    '0.78': [1792.0, 2304.0], '0.82': [1792.0, 2176.0], '0.88': [1920.0, 2176.0], '0.94': [1920.0, 2048.0],
    '1.0': [2048.0, 2048.0], '1.07': [2048.0, 1920.0], '1.13': [2176.0, 1920.0], '1.21': [2176.0, 1792.0],
    '1.29': [2304.0, 1792.0], '1.38': [2304.0, 1664.0], '1.46': [2432.0, 1664.0], '1.67': [2560.0, 1536.0],
    '1.75': [2688.0, 1536.0], '2.0': [2816.0, 1408.0], '2.09': [2944.0, 1408.0], '2.4': [3072.0, 1280.0],
    '2.5': [3200.0, 1280.0], '2.89': [3328.0, 1152.0], '3.0': [3456.0, 1152.0], '3.11': [3584.0, 1152.0],
    '3.62': [3712.0, 1024.0], '3.75': [3840.0, 1024.0], '3.88': [3968.0, 1024.0], '4.0': [4096.0, 1024.0]
}

ASPECT_RATIO_1024 = {
    '0.25': [512., 2048.], '0.26': [512., 1984.], '0.27': [512., 1920.], '0.28': [512., 1856.],
    '0.32': [576., 1792.], '0.33': [576., 1728.], '0.35': [576., 1664.], '0.4':  [640., 1600.],
    '0.42':  [640., 1536.], '0.48': [704., 1472.], '0.5': [704., 1408.], '0.52': [704., 1344.],
    '0.57': [768., 1344.], '0.6': [768., 1280.], '0.68': [832., 1216.], '0.72': [832., 1152.],
    '0.78': [896., 1152.], '0.82': [896., 1088.], '0.88': [960., 1088.], '0.94': [960., 1024.],
    '1.0':  [1024., 1024.], '1.07': [1024.,  960.], '1.13': [1088.,  960.], '1.21': [1088.,  896.],
    '1.29': [1152.,  896.], '1.38': [1152.,  832.], '1.46': [1216.,  832.], '1.67': [1280.,  768.],
    '1.75': [1344.,  768.], '2.0':  [1408.,  704.], '2.09':  [1472.,  704.], '2.4':  [1536.,  640.],
    '2.5':  [1600.,  640.], '2.89':  [1664.,  576.], '3.0':  [1728.,  576.], '3.11':  [1792.,  576.],
    '3.62':  [1856.,  512.], '3.75':  [1920.,  512.], '3.88':  [1984.,  512.], '4.0':  [2048.,  512.],
}

ASPECT_RATIO_512 = {
     '0.25': [256.0, 1024.0], '0.26': [256.0, 992.0], '0.27': [256.0, 960.0], '0.28': [256.0, 928.0],
     '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
     '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
     '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
     '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
     '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
     '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
     '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
     '2.5': [800.0, 320.0], '2.89': [832.0, 288.0], '3.0': [864.0, 288.0], '3.11': [896.0, 288.0],
     '3.62': [928.0, 256.0], '3.75': [960.0, 256.0], '3.88': [992.0, 256.0], '4.0': [1024.0, 256.0]
     }

ASPECT_RATIO_256 = {
     '0.25': [128.0, 512.0], '0.26': [128.0, 496.0], '0.27': [128.0, 480.0], '0.28': [128.0, 464.0],
     '0.32': [144.0, 448.0], '0.33': [144.0, 432.0], '0.35': [144.0, 416.0], '0.4': [160.0, 400.0],
     '0.42': [160.0, 384.0], '0.48': [176.0, 368.0], '0.5': [176.0, 352.0], '0.52': [176.0, 336.0],
     '0.57': [192.0, 336.0], '0.6': [192.0, 320.0], '0.68': [208.0, 304.0], '0.72': [208.0, 288.0],
     '0.78': [224.0, 288.0], '0.82': [224.0, 272.0], '0.88': [240.0, 272.0], '0.94': [240.0, 256.0],
     '1.0': [256.0, 256.0], '1.07': [256.0, 240.0], '1.13': [272.0, 240.0], '1.21': [272.0, 224.0],
     '1.29': [288.0, 224.0], '1.38': [288.0, 208.0], '1.46': [304.0, 208.0], '1.67': [320.0, 192.0],
     '1.75': [336.0, 192.0], '2.0': [352.0, 176.0], '2.09': [368.0, 176.0], '2.4': [384.0, 160.0],
     '2.5': [400.0, 160.0], '2.89': [416.0, 144.0], '3.0': [432.0, 144.0], '3.11': [448.0, 144.0],
     '3.62': [464.0, 128.0], '3.75': [480.0, 128.0], '3.88': [496.0, 128.0], '4.0': [512.0, 128.0]
}

ASPECT_RATIO_256_TEST = {
     '0.25': [128.0, 512.0], '0.28': [128.0, 464.0],
     '0.32': [144.0, 448.0], '0.33': [144.0, 432.0], '0.35': [144.0, 416.0], '0.4': [160.0, 400.0],
     '0.42': [160.0, 384.0], '0.48': [176.0, 368.0], '0.5': [176.0, 352.0], '0.52': [176.0, 336.0],
     '0.57': [192.0, 336.0], '0.6': [192.0, 320.0], '0.68': [208.0, 304.0], '0.72': [208.0, 288.0],
     '0.78': [224.0, 288.0], '0.82': [224.0, 272.0], '0.88': [240.0, 272.0], '0.94': [240.0, 256.0],
     '1.0': [256.0, 256.0], '1.07': [256.0, 240.0], '1.13': [272.0, 240.0], '1.21': [272.0, 224.0],
     '1.29': [288.0, 224.0], '1.38': [288.0, 208.0], '1.46': [304.0, 208.0], '1.67': [320.0, 192.0],
     '1.75': [336.0, 192.0], '2.0': [352.0, 176.0], '2.09': [368.0, 176.0], '2.4': [384.0, 160.0],
     '2.5': [400.0, 160.0], '3.0': [432.0, 144.0],
     '4.0': [512.0, 128.0]
}

ASPECT_RATIO_512_TEST = {
     '0.25': [256.0, 1024.0], '0.28': [256.0, 928.0],
     '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
     '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
     '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
     '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
     '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
     '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
     '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
     '2.5': [800.0, 320.0], '3.0': [864.0, 288.0],
     '4.0': [1024.0, 256.0]
     }

ASPECT_RATIO_1024_TEST = {
    '0.25': [512., 2048.], '0.28': [512., 1856.],
    '0.32': [576., 1792.], '0.33': [576., 1728.], '0.35': [576., 1664.], '0.4':  [640., 1600.],
    '0.42':  [640., 1536.], '0.48': [704., 1472.], '0.5': [704., 1408.], '0.52': [704., 1344.],
    '0.57': [768., 1344.], '0.6': [768., 1280.], '0.68': [832., 1216.], '0.72': [832., 1152.],
    '0.78': [896., 1152.], '0.82': [896., 1088.], '0.88': [960., 1088.], '0.94': [960., 1024.],
    '1.0':  [1024., 1024.], '1.07': [1024.,  960.], '1.13': [1088.,  960.], '1.21': [1088.,  896.],
    '1.29': [1152.,  896.], '1.38': [1152.,  832.], '1.46': [1216.,  832.], '1.67': [1280.,  768.],
    '1.75': [1344.,  768.], '2.0':  [1408.,  704.], '2.09':  [1472.,  704.], '2.4':  [1536.,  640.],
    '2.5':  [1600.,  640.], '3.0':  [1728.,  576.],
    '4.0':  [2048.,  512.],
}

ASPECT_RATIO_2048_TEST = {
    '0.25': [1024.0, 4096.0], '0.26': [1024.0, 3968.0],
    '0.32': [1152.0, 3584.0], '0.33': [1152.0, 3456.0], '0.35': [1152.0, 3328.0], '0.4': [1280.0, 3200.0],
    '0.42': [1280.0, 3072.0], '0.48': [1408.0, 2944.0], '0.5': [1408.0, 2816.0], '0.52': [1408.0, 2688.0],
    '0.57': [1536.0, 2688.0], '0.6': [1536.0, 2560.0], '0.68': [1664.0, 2432.0], '0.72': [1664.0, 2304.0],
    '0.78': [1792.0, 2304.0], '0.82': [1792.0, 2176.0], '0.88': [1920.0, 2176.0], '0.94': [1920.0, 2048.0],
    '1.0': [2048.0, 2048.0], '1.07': [2048.0, 1920.0], '1.13': [2176.0, 1920.0], '1.21': [2176.0, 1792.0],
    '1.29': [2304.0, 1792.0], '1.38': [2304.0, 1664.0], '1.46': [2432.0, 1664.0], '1.67': [2560.0, 1536.0],
    '1.75': [2688.0, 1536.0], '2.0': [2816.0, 1408.0], '2.09': [2944.0, 1408.0], '2.4': [3072.0, 1280.0],
    '2.5': [3200.0, 1280.0], '3.0': [3456.0, 1152.0],
    '4.0': [4096.0, 1024.0]
}

ASPECT_RATIO_2880_TEST = {
    '0.25': [2048.0, 8192.0], '0.26': [2048.0, 7936.0],
    '0.32': [2304.0, 7168.0], '0.33': [2304.0, 6912.0], '0.35': [2304.0, 6656.0], '0.4': [2560.0, 6400.0],
    '0.42': [2560.0, 6144.0], '0.48': [2816.0, 5888.0], '0.5': [2816.0, 5632.0], '0.52': [2816.0, 5376.0],
    '0.57': [3072.0, 5376.0], '0.6': [3072.0, 5120.0], '0.68': [3328.0, 4864.0], '0.72': [3328.0, 4608.0],
    '0.78': [3584.0, 4608.0], '0.82': [3584.0, 4352.0], '0.88': [3840.0, 4352.0], '0.94': [3840.0, 4096.0],
    '1.0': [4096.0, 4096.0], '1.07': [4096.0, 3840.0], '1.13': [4352.0, 3840.0], '1.21': [4352.0, 3584.0],
    '1.29': [4608.0, 3584.0], '1.38': [4608.0, 3328.0], '1.46': [4864.0, 3328.0], '1.67': [5120.0, 3072.0],
    '1.75': [5376.0, 3072.0], '2.0': [5632.0, 2816.0], '2.09': [5888.0, 2816.0], '2.4': [6144.0, 2560.0],
    '2.5': [6400.0, 2560.0], '3.0': [6912.0, 2304.0],
    '4.0': [8192.0, 2048.0],
}

def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
