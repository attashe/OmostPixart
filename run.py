import os
os.environ['FORCE_MEM_EFFICIENT_ATTN'] = "1"
os.environ['XDG_CACHE_HOME'] = 'K:/Weights/'
import lib_omost.memory_management as memory_management

import torch
import numpy as np
import gradio as gr

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
import torch

from threading import Thread

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import CLIPTextModel, CLIPTokenizer
from lib_omost.pipeline import StableDiffusionXLOmostPipeline
# from chat_interface import ChatInterface
from transformers.generation.stopping_criteria import StoppingCriteriaList

import lib_omost.canvas as omost_canvas
from pixart_utils import Text2ImageModel, OmostLLM


class Context:
    llm_model = None
    t2i_model = None
    
    # omost prompt cache
    omost_prompt_cache = {}

def generate_image(prompt, cfg, steps, aspect_ratio, seed, 
                   normalization_type, cfg_schedule_type, shrink_prompt,
                   scale_to_one, negative_rescale):
    # Debug
    print("Prompt: ", prompt)
    print("Steps: ", steps)
    print("Aspect Ratio: ", aspect_ratio)
    print("Seed: ", seed)
    print("Normalization Type: ", normalization_type)
    print("CFG Schedule Type: ", cfg_schedule_type)
    print("Shrink Prompt: ", shrink_prompt)
    print("Scale to One: ", scale_to_one)
    print("Negative Rescale: ", negative_rescale)
    
    # image = Image.new('RGB', (256, 256), "blue")
    
    # change height and width in aspect ratio
    aspect_ratio = aspect_ratio.split(":")
    aspect_ratio = f'{aspect_ratio[0]}:{aspect_ratio[1]}'
    
    # DONE: Implement prompt cache
    # TODO: Add memory management for omost llm model
    # TODO: Add button for generate omost prompt seprate from image generation
    
    # clean prompt
    strip_prompt = prompt.strip()
    # check if prompt is in cache
    if strip_prompt in Context.omost_prompt_cache:
        omost_prompt = Context.omost_prompt_cache[strip_prompt]
    else:
        omost_prompt = Context.llm_model.inference(prompt)
        Context.omost_prompt_cache[strip_prompt] = omost_prompt

    image = Context.t2i_model.inference(
        omost_prompt,
        sampler="dpm-solver", 
        cfg=cfg, steps=steps, aspect_ratio=aspect_ratio, seed=seed,
        normalization_type=normalization_type, cfg_schedule_type=cfg_schedule_type,
        scale_to_one=scale_to_one, negative_rescale=negative_rescale,
        shrink_prompt=shrink_prompt)
    
    return image


def random_seed():
    return np.random.randint(0, 2**31 - 1)


def main():
    Context.llm_model = OmostLLM()
    # Context.llm_model.test_inference()
    
    Context.t2i_model = Text2ImageModel()
    # Context.t2i_model.test_inference()
    
    with gr.Blocks() as demo:
        gr.Markdown("# Image Generation Demo")

        with gr.Box():
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt")
                    cfg = gr.Slider(1.1, 15, value=3.5, step=0.1, label="CFG (Classifier-Free Guidance)")
                    steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                    aspect_ratio = gr.Radio(
                        ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "2:1", "1:2"],
                        label="Aspect Ratio", value="1:1")
                    seed_mode = gr.Radio(["Random Seed", "Given Seed"], label="Seed Mode", value="Random Seed")
                    seed = gr.Number(value=random_seed(), label="Seed")
                    normalization_type = gr.Radio(
                        ["First", "Second", "Second v2", "Second v3", "Third"],
                        value="Second v2",
                        label="Masking Normalization Type")
                    cfg_schedule_type = gr.Radio(["Constant", "Linear", "Cosine"], label="CFG Schedule Type", value="None")
                    shrink_prompt = gr.Checkbox(label="Shrink Prompt", value=False)
                    scale_to_one = gr.Checkbox(label="Scale to One", value=True)
                    negative_rescale = gr.Checkbox(label="Negative Rescale", value=False)

                    def update_seed_mode(seed_mode):
                        return gr.update(visible=seed_mode == "Given Seed")

                    seed_mode.change(fn=update_seed_mode, inputs=seed_mode, outputs=seed)

                    generate_button = gr.Button("Generate Image")
                with gr.Column():
                    image_output = gr.Image(label="Output", elem_id="output-img")

        generate_button.click(
            fn=generate_image,
            inputs=[prompt, cfg, steps, aspect_ratio, seed, 
                    normalization_type, cfg_schedule_type, shrink_prompt,
                    scale_to_one, negative_rescale],
            outputs=[image_output]
        )

    demo.launch(share=True, enable_queue=True)


if __name__ == "__main__":
    main()