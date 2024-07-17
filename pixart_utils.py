import os
os.environ['FORCE_MEM_EFFICIENT_ATTN'] = "1"
os.environ['XDG_CACHE_HOME'] = 'K:/Weights/'

import numpy as np
from typing import Union, List
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid
from diffusers.models import AutoencoderKL
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import randn_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from diffusion.dpms import DPMS
from diffusion.sa_solver import SASolverSampler
from diffusion.pixart_ms import PixArtMS_XL_2
from diffusion.t5 import T5Embedder
from diffusion.utils import ASPECT_RATIO_1024_TEST
from diffusion.utils import prepare_prompt_ar, mask_feature

from lib_omost.canvas import Canvas


device = "cuda" if torch.cuda.is_available() else "cpu"

class args:
    image_size = 1024
    bs = 1
    cfg_scale = 4.0
    seed = 0
    step = -1
    version: str = 'sigma'
    sampling_algo = 'dpms'
    model_path: str = "K:/Weights/PixArtXL/PixArt-Sigma-XL-2-1024-MS.pth"
    base_ratios = eval(f'ASPECT_RATIO_{image_size}_TEST')


class Text2ImageModel:
    
    def __init__(self):
        self.model = None
        self.llm_embed_model = None
        self.vae = None
        self.model_loaded = False

    def load_model(self):
        if self.model is not None \
            and self.llm_embed_model is not None \
            and self.vae is not None:
            
            if self.model_loaded:
                self.model.to(device)
                # self.llm_embed_model.to(device)
                self.vae.to(device)
                self.model_loaded = True
            
            return
        
        t5_path = 'K:/Weights/'
        self.llm_embed_model = T5Embedder(
            device=torch.device('cuda'), cache_dir=t5_path,
            # use_mixed_device=True,
            use_offload_folder='K:/Weights/offload/',
            torch_dtype=torch.float16, model_max_length=300)

        assert args.image_size in [256, 512, 1024, 2048], \
            "We only provide pre-trained models for 256x256, 512x512, 1024x1024 and 2048x2048 resolutions."
        pe_interpolation = {256: 0.5, 512: 1, 1024: 2, 2048: 4}
        latent_size = args.image_size // 8
        max_sequence_length = {"alpha": 120, "sigma": 300}[args.version]
        weight_dtype = torch.float16
        micro_condition = True if args.version == 'alpha' and args.image_size == 1024 else False
        if args.image_size in [512, 1024, 2048, 2880]:
            self.model = PixArtMS_XL_2(
                input_size=latent_size,
                pe_interpolation=pe_interpolation[args.image_size],
                micro_condition=micro_condition,
                model_max_length=max_sequence_length,
            ).to(device)
            
        # print("Generating sample from ckpt: %s" % args.model_path)
        state_dict = find_model(args.model_path)
        if 'pos_embed' in state_dict['state_dict']:
            del state_dict['state_dict']['pos_embed']
        missing, unexpected = self.model.load_state_dict(state_dict['state_dict'], strict=False)
        print('Missing keys: ', missing)
        print('Unexpected keys', unexpected)
        self.model.eval()
        self.model.to(weight_dtype)
        
        self.vae = AutoencoderKL.from_pretrained("K:/Weights/PixArtXL/diffusers/vae").eval().to(device).to(weight_dtype)

        self.model_loaded = True

    def unload_model(self):
        if not self.model_loaded:
            return
        self.model.to('cpu')
        # self.llm_embed_model.to('cpu')
        self.vae.to('cpu')

    def test_inference(self):
        if self.model is None or self.llm_embed_model is None or self.vae is None:
            self.load_model()
        
        mask_pil = Image.open('K:/Images/storia/product_photo/cola_mask.png')
        mask2_pil = ImageOps.mirror(mask_pil)
        
        negative_prompt = 'blurry'
        prompt_base = "Photo of table"
        prompt = prompt_base + ' --ar 2:3'
        seed = 1
        sampler = "dpm-solver"
        img, cross_mask = generate_img_regions(
            self.model, self.llm_embed_model, self.vae,
            prompt, 
            sampler, 20, 10, negative_prompt=negative_prompt, seed=seed, dtype=torch.float16,
            reg_prompts=[
                'photo of red cola can',
                'photo of blue bottle'
            ],
            masks=[
                mask2_pil,
                mask_pil,
            ],
            scale_type='second',
            )
        Image.fromarray(img).save('inference_test.png')
        
        self.unload_model()
    
    # boxes: List[List[int]]
    def inference_bbox(self, prompt: str, negative_prompt: str, masks: List[Image.Image], sub_prompts: List[str],
                       aspect_ratio: str, seed: int,
                       cfg: float = 3.5, steps: int = 20,
                       normalization_type="first", cfg_schedule_type="constant",
                       scale_to_one=False, negative_rescale=False,):
        # Second pass is not working
        if self.model is None or self.llm_embed_model is None or self.vae is None:
            self.load_model()
        
        # Calculate w/h from aspect ratio
        prompt_ar = prompt + f' --ar {aspect_ratio}'
        print('Prompt AR:', prompt_ar)
        # prompt_clean, prompt_show, hw, ar, custom_hw = prepare_prompt_ar(prompt_ar, args.base_ratios, device=device)      # ar for aspect ratio
        # height, width = hw[0, 0], hw[0, 1]
        
        # Build masks from list of boxes
        # masks = []
        # for box in boxes:
        #     mask = Image.new('L', (width, height), 0)
        #     mask.paste(255, box)
        #     masks.append(mask)

        sampler = "dpm-solver"
        img, _ = generate_img_regions(
            self.model, self.llm_embed_model, self.vae,
            prompt_ar,
            sampler, steps, cfg, negative_prompt=negative_prompt, seed=seed, dtype=torch.float16,
            reg_prompts=sub_prompts,
            masks=masks,
            scale_type='second_rescale',
            scale_to_one=True,
            negative_rescale=False,
            cfg_schedule_type='linear',
            )
        Image.fromarray(img).save('inference_test_gligen.png')
        
        # self.unload_model()
        
        return img
        
    def build_prompt(self, canvas_outputs, shrink_prompt=False):
        subprompts_list = []

        if not shrink_prompt:
            for i, cond_dict in enumerate(canvas_outputs['bag_of_conditions']):
                cond_prompt = ''
                if i == 0:
                    cond_prompt += ' '.join(cond_dict['prefixes'])
                    cond_prompt += ' '.join(cond_dict['suffixes'])
                else:
                    cond_prompt += ' '.join(cond_dict['prefixes'][-1])
                    cond_prompt += ' '.join(cond_dict['suffixes'])
                
                subprompts_list.append({
                    'prompt': cond_prompt,
                    'mask': Image.fromarray(cond_dict['mask'].astype(np.uint8) * 255)
                })
        else:
            for i, cond_dict in enumerate(canvas_outputs['bag_of_conditions']):
                cond_prompt = ''
                if i == 0:
                    cond_prompt += ' '.join(cond_dict['prefixes'])
                    cond_prompt += ' '.join(cond_dict['suffixes'])
                else:
                    cond_prompt += ' '.join(cond_dict['prefixes'][-1])
                    cond_prompt += ' '.join(cond_dict['suffixes'][:2])
                
                subprompts_list.append({
                    'prompt': cond_prompt,
                    'mask': Image.fromarray(cond_dict['mask'].astype(np.uint8) * 255)
                })
                
        return subprompts_list

    def inference(self, prompt, sampler="dpm-solver", 
                  cfg=3.5, steps=2, aspect_ratio="1:1", seed=0,
                  normalization_type="first", cfg_schedule_type="constant",
                  scale_to_one=False, negative_rescale=False,
                  shrink_prompt=False):
        if self.model is None or self.llm_embed_model is None or self.vae is None:
            self.load_model()
        
        canvas = Canvas.from_bot_response(prompt)
        canvas_outputs = canvas.process()
        subprompts_list = self.build_prompt(canvas_outputs, shrink_prompt)
        
        negative_prompt = 'blurry'
        prompt_base = subprompts_list[0]['prompt']
        prompt = prompt_base + f' --ar {aspect_ratio}'
        print('Base prompt:', prompt)
        cfg_schedule_type = cfg_schedule_type.lower()
        assert sampler in ['dpm-solver', 'sa-solver']
        assert cfg_schedule_type in ['constant', 'linear', 'cosine']
        
        if normalization_type == 'First':
            normalization_type = 'first'
        elif normalization_type == 'Second':
            normalization_type = 'second'
        elif normalization_type == 'Second v2':
            normalization_type = 'second_onemin_add'
        elif normalization_type == 'Second v3':
            normalization_type = 'second_rescale'
        elif normalization_type == 'Third':
            normalization_type = 'third'

        img, cross_mask = generate_img_regions(
            self.model, self.llm_embed_model, self.vae,
            prompt, sampler, steps, cfg,
            negative_prompt=negative_prompt,
            seed=seed, dtype=torch.float16,
            reg_prompts=[sp['prompt'] for sp in subprompts_list[1:]],
            masks=[sp['mask'] for sp in subprompts_list[1:]],
            cfg_schedule_type=cfg_schedule_type,
            scale_type=normalization_type,
            scale_to_one=scale_to_one,
            negative_rescale=negative_rescale,
        )
        output_img = Image.fromarray(img)
        output_img.save('inference_test.png')
        
        # FIXME: Unload model after inference
        # self.unload_model()

        return output_img

class OmostLLM:
    
    def __init__(self):
        self.llm_model = None
        self.llm_tokenizer = None
        
        self.llm_model, self.llm_tokenizer = self.load_llm()
        
        self.system_prompt = r'''You are a helpful AI assistant to compose images using the below python class `Canvas`:

```python
class Canvas:
    def set_global_description(self, description: str, detailed_descriptions: list[str], tags: str, HTML_web_color_name: str):
        pass

    def add_local_description(self, location: str, offset: str, area: str, distance_to_viewer: float, description: str, detailed_descriptions: list[str], tags: str, atmosphere: str, style: str, quality_meta: str, HTML_web_color_name: str):
        assert location in ["in the center", "on the left", "on the right", "on the top", "on the bottom", "on the top-left", "on the top-right", "on the bottom-left", "on the bottom-right"]
        assert offset in ["no offset", "slightly to the left", "slightly to the right", "slightly to the upper", "slightly to the lower", "slightly to the upper-left", "slightly to the upper-right", "slightly to the lower-left", "slightly to the lower-right"]
        assert area in ["a small square area", "a small vertical area", "a small horizontal area", "a medium-sized square area", "a medium-sized vertical area", "a medium-sized horizontal area", "a large square area", "a large vertical area", "a large horizontal area"]
        assert distance_to_viewer > 0
        pass
```'''
    
    def load_llm(self, model_base='llama3'):
        if model_base == 'llama3':
            model_name = 'lllyasviel/omost-llama-3-8b-4bits'
        elif model_base == 'phi3':
            model_name = 'lllyasviel/omost-phi-3-mini-128k'
        elif model_base == 'dolphin':
            model_name = 'lllyasviel/omost-dolphin-2.9-llama3-8b'
        
        print(f'Loading LLM model {model_name}')
        
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        #     token=HF_TOKEN,
            device_map="auto",
            trust_remote_code=True,
        )

        llm_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        #     token=HF_TOKEN
        )
        
        return llm_model, llm_tokenizer

    @torch.inference_mode()
    def test_inference(self):
        # self.load_model()

        message = 'Draw a very british laptop'
        
        conversation = [{"role": "system", "content": self.system_prompt}]
        conversation.append({"role": "user", "content": message})

        input_ids = self.llm_tokenizer.apply_chat_template(
            conversation, return_tensors="pt", add_generation_prompt=True).to(self.llm_model.device)

        # streamer = TextIteratorStreamer(llm_tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        streamer = TextStreamer(self.llm_tokenizer)
        
        max_new_tokens = 1024
        temperature = 0.5
        top_p = 0.9

        generate_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
        #     stopping_criteria=stopping_criteria,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        if temperature == 0:
            generate_kwargs['do_sample'] = False
            
        print(self.llm_model.generate(**generate_kwargs))
        
    @torch.inference_mode()
    def inference(self, prompt):
        conversation = [{"role": "system", "content": self.system_prompt}]
        conversation.append({"role": "user", "content": prompt})

        input_ids = self.llm_tokenizer.apply_chat_template(
            conversation, return_tensors="pt", add_generation_prompt=True).to(self.llm_model.device)
        
        max_new_tokens = 4096
        temperature = 0.5
        top_p = 0.9

        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        if temperature == 0:
            generate_kwargs['do_sample'] = False
            
        output_tks = self.llm_model.generate(**generate_kwargs)
        text = self.llm_tokenizer.decode(output_tks[0], skip_special_tokens=True)
        text = text[len(self.system_prompt) + len(prompt):]
        print(f'Generated text: {text}')
        
        return text
    
    def load_model(self):
        if self.llm_model is None or self.llm_tokenizer is None:
            self.llm_model, self.llm_tokenizer = self.load_llm()
        
        self.llm_model.to(device)
        self.llm_tokenizer.to(device)
    
    def unload_llm(self):
        if self.llm_model is not None:
            self.llm_model.to('cpu')
        if self.llm_tokenizer is not None:
            self.llm_tokenizer.to('cpu')
    

@torch.no_grad()
def ndarr_image(tensor: Union[torch.Tensor, List[torch.Tensor]], **kwargs,) -> None:
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return ndarr

def find_model(model_name):
    assert os.path.isfile(model_name), f'Could not find PixArt checkpoint at {model_name}'
    return torch.load(model_name, map_location=lambda storage, loc: storage)

def resize_img(samples, hw, custom_hw):
    if (hw != custom_hw).all():
        if custom_hw[0,0] / hw[0,0] > custom_hw[0,1] / hw[0,1]:
            resize_size = int(custom_hw[0,0]), int(hw[0,1] * custom_hw[0,0] / hw[0,0])
        elif custom_hw[0,0] / hw[0,0] > custom_hw[0,1] / hw[0,1]:
            resize_size = int(hw[0,0] * custom_hw[0,1] / hw[0,1]), int(custom_hw[0,1])
        else:
            resize_size = int(custom_hw[0,0]), int(custom_hw[0,1])
        transform = T.Compose([
        T.Resize(resize_size),  # Image.BICUBIC
        T.CenterCrop(resize_size),
        ])
        return transform(samples)
    else:
        return samples
    
def encode(llm_embed_model, prompt, dtype):
    prompt_clean = prompt.strip()
    if isinstance(prompt_clean, str):
        prompts = [prompt_clean]

    caption_embs, emb_masks = llm_embed_model.get_text_embeddings(prompts)
    caption_embs = caption_embs[:, None]
    masked_embs, keep_index = mask_feature(caption_embs, emb_masks)
    masked_embs = masked_embs.cuda().to(dtype)

    return caption_embs, masked_embs, keep_index


@torch.inference_mode()
def generate_img_regions(model, llm_embed_model, vae, prompt, sampler, sample_steps, scale, 
                 negative_prompt=None, latent=None,
                 seed=42, dtype=torch.float32,
                 reg_prompts=None, masks=None, scale_type='constant', scale_to_one=False,
                 negative_rescale=False,
                 cfg_schedule_type='None',
                       ):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    cross_attn_kwargs = {}
    
    prompt_clean, prompt_show, hw, ar, custom_hw = prepare_prompt_ar(prompt, args.base_ratios, device=device)      # ar for aspect ratio
    prompt_clean = prompt_clean.strip()
    if isinstance(prompt_clean, str):
        prompts = [prompt_clean]

    caption_embs, emb_masks = llm_embed_model.get_text_embeddings(prompts)
    caption_embs = caption_embs[:, None]
    masked_embs, keep_index = mask_feature(caption_embs, emb_masks)
    masked_embs = masked_embs.cuda().to(dtype)

    print(f'{caption_embs.shape=}, {masked_embs.shape=}')
    
    null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None].to(dtype)
    
    if negative_prompt:
        batch_size = len(prompts)
        uncond_tokens = [negative_prompt] * batch_size
        neg_caption_embs, neg_emb_masks = llm_embed_model.get_text_embeddings(uncond_tokens)
        neg_caption_embs = neg_caption_embs[:, None]
        neg_masked_embs, neg_keep_index = mask_feature(neg_caption_embs, neg_emb_masks)
        
        null_y[:, :, :neg_keep_index] = neg_masked_embs
#         neg_keep_index = keep_index
    
    print("NEG KEEP ", neg_keep_index)
    print(null_y[:, :, :neg_keep_index, :].shape)
    if reg_prompts:
        reg_embeds = []
        for p in reg_prompts:
            red_emb, reg_emb_masked, length = encode(llm_embed_model, p, dtype=dtype)
            print(f'{reg_emb_masked.shape=}')
            reg_embeds.append(reg_emb_masked)
        
        masked_embs = torch.cat([masked_embs,] + reg_embeds, dim=2)
        print(f'{masked_embs.shape=}')
        
        hH, hW = int(hw[0, 0]) // 16, int(hw[0, 1]) // 16
        print(hw, '->', hH, hW)
        lin_masks = []
        for mask in masks:
            mask = mask.convert('L')
            mask = torch.tensor(np.array(mask)).unsqueeze(0).unsqueeze(0) / 255
            print(f'Mask size {mask.shape} -> ', end='')
            mask = torch.nn.functional.interpolate(mask, (hH, hW), mode='nearest-exact').flatten()
            print(f'{mask.shape}')
            # Linearize mask
            lin_masks.append(mask)
        
        Ny = 1152
        Nx = int(hH * hW)
        
        emb_len = masked_embs.shape[2]
#         neg_keep_index = emb_len
    
        print(Nx, emb_len, neg_keep_index)
        cross_mask = torch.zeros(Nx*2, 
                                 emb_len + neg_keep_index)
        print(f'{cross_mask.shape=}')
        # cross_mask.fill_(float('-inf'))
        cross_mask[:Nx, :keep_index] = 1

        emb_cum_idx = keep_index
        uncond_mask = torch.zeros(Nx)

        for m, emb in zip(lin_masks, reg_embeds):
            # Extend mask to uncond part
            m = torch.cat([m, uncond_mask])
            mb = m > 0.5
            cross_mask[mb, emb_cum_idx : emb_cum_idx + emb.shape[2]] = 1
            emb_cum_idx += emb.shape[2]
        
        lin_masks = torch.stack(lin_masks, dim=0)
        print(f'{lin_masks.shape=}')
        # Uncond masking
        cross_mask[Nx:, -neg_keep_index:] = 1
        
        num_heads = 16
        
        cross_mask_extended = cross_mask.unsqueeze(0).unsqueeze(0).repeat(1, num_heads, 1, 1).to(device)
        print(f'{cross_mask_extended.shape=}')
        cross_attn_kwargs = {
            'region_mask': cross_mask_extended,
            'lin_masks': lin_masks,
            'mask_scale_type': scale_type,
            'scale_to_one': scale_to_one,
            'negative_rescale': negative_rescale,
            'cfg_schedule': cfg_schedule_type,
        }
        print(f"{cross_attn_kwargs['region_mask'].shape}")
#         return lin_masks

    latent_size_h, latent_size_w = int(hw[0, 0]//8), int(hw[0, 1]//8)
    # Sample images:
    if sampler == 'dpm-solver':
        with torch.autocast("cuda"):
            # Create sampling noise:
            n = len(prompts)
            if latent is not None:
                z = torch.nn.functional.interpolate(latent, (latent_size_h, latent_size_w), mode='nearest-exact')
                z = z.to(dtype).to(device)
            else:
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device, dtype=dtype)
            model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar},
                               cross_attn_kwargs=cross_attn_kwargs)
            print(z.dtype, null_y.dtype, masked_embs.dtype)
            dpm_solver = DPMS(model.forward_with_dpmsolver,
                              condition=masked_embs.to(device),
                              uncondition=null_y[:, :, :neg_keep_index, :].to(device),
                              cfg_scale=scale,
                              model_kwargs=model_kwargs)
        
            samples = dpm_solver.sample(
                z,
                steps=sample_steps,
                order=2,
                skip_type="time_uniform",
                method="multistep",
            )
    
    torch.cuda.empty_cache()

    samples = samples.to(vae.dtype)
    samples = vae.decode(samples / vae.config.scaling_factor).sample
    torch.cuda.empty_cache()
    samples = resize_img(samples, hw, custom_hw)
    
    out_img = ndarr_image(samples, normalize=True, value_range=(-1, 1))
    
    return out_img, cross_mask_extended