import io
import json
import flask

import urllib.request
import urllib.parse
import urllib.error
import numpy as np

from pathlib import Path
from flask import Flask, send_file
from flask_socketio import SocketIO, send, emit
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

import lib_omost.canvas as omost_canvas
from pixart_utils import Text2ImageModel, OmostLLM
from threading import Thread, Lock
from flux.generate import FluxInference

app = Flask(__name__)
socketio = SocketIO(app)

    
class ImageStorage:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.images = {}
        # Save image order for LRU
        self.order = []
    
    def __len__(self):
        return len(self.images)
    
    def __contains__(self, key):
        return key in self.images
    
    def __getitem__(self, key):
        return self.images[key]
    
    def __setitem__(self, key, value):
        if len(self.images) >= self.capacity:
            # Remove the least recently used image
            del self.images[self.order[0]]
            self.order.pop(0)
        self.images[key] = value
        self.order.append(key)
        
    def __delitem__(self, key):
        del self.images[key]
        self.order.remove(key)
        
    def clear(self):
        self.images.clear()


class Context:
    image_id: int = 1
    # t2i_model: Text2ImageModel = None
    t2i_model: FluxInference = None
    image_storage = ImageStorage(50)
    lock = Lock()


def generate_image_wrapper(payload):
    
    with Context.lock:
        bboxes = payload['prompt']['bboxes']
        masks = []
        subprompts = []
        
        for bbox in bboxes:
            mask = Image.new('L', (payload['prompt']['width'], payload['prompt']['height']), 0)
            mask_arr = np.array(mask)
            
            # Draw the bounding box
            mask_arr[bbox['y']:bbox['y']+bbox['height'], bbox['x']:bbox['x']+bbox['width']] = 255
            
            mask = Image.fromarray(mask_arr)
            
            # Debug save the mask
            mask.save(f'mask_{bbox["idx"]}.png')
            
            masks.append(mask)
            subprompts.append(bbox['caption'])
            
        # Mock the image
        # image = Image.new('RGB', (payload['prompt']['width'], payload['prompt']['height']), "blue")
        
        # if Context.t2i_model is None:
            # Context.t2i_model = Text2ImageModel()        
        
        image = Context.t2i_model.inference_bbox(
            prompt=payload['prompt']['positive'], negative_prompt=payload['prompt']['negative'],
            masks=masks, subprompts=subprompts,
            aspect_ratio='1:1', seed=int(payload['prompt']['seed']), steps=int(payload['prompt']['steps']),
            guidance=float(payload['prompt']['cfg']),
            height=payload['prompt']['height'], width=payload['prompt']['width']
        )
        
        Context.image_storage[Context.image_id] = Image.fromarray(image)
    
    # Create a folder for the outputs
    Path(f'outputs').mkdir(parents=True, exist_ok=True)
    
    # Save the image with the image ID and all payload as metadata
    metadata = PngInfo()
    metadata.add_text('payload', json.dumps(payload))
    image = Image.fromarray(image)
    image.save(f'image_{Context.image_id}.png', pnginfo=metadata)
    
    # Save payload to a file
    with open(f'payload_{Context.image_id}.json', 'w') as f:
        json.dump(payload, f)
    
    # emit('image_generated', {'image_id': Context.image_id})


@app.route("/")
def index():
    print('Load index page')
    return flask.render_template('base.html')

@app.route('/generate_image', methods=['GET', 'POST'])
def generate_image():
    # Get a payload from the request
    payload = flask.request.get_json()
    print(f'Payload: {payload}')
    
    # Generate the image
    # image = generate_image_pixart(
    #     prompt="a painting of a nerdy rodent",
    #     cfg="omost",
    #     steps=20,
    #     aspect_ratio="16:9",
    #     seed=42,
    #     normalization_type="default",
    #     cfg_schedule_type="default",
    #     shrink_prompt=False,
    #     scale_to_one=False,
    #     negative_rescale=False
    # )
    
    # JS code
    # prompt['bboxes'][String(box_idx)] = {
    #   'caption': box.caption,
    #   'width': box.width,
    #   'height': box.height,
    #   'x': box.x,
    #   'y': box.y,
    #   'idx': box_idx,
    # };
    
    # thread = Thread(target=generate_image_wrapper, args=(payload,), daemon=True)
    # thread.start()
    # thread.join()
    generate_image_wrapper(payload)
    
    print(f'Image ID: {Context.image_id}')
    # emit('image_generated', {'image_id': Context.image_id}, namespace='/')
    
    response = json.dumps({'image_id':Context.image_id})
    Context.image_id += 1
    
    return response, 200, {'ContentType':'application/json'}

@app.route('/get_image/<input_image_id>', methods=['GET'])
def get_image(input_image_id):
    # Get the image
    image = Context.image_storage[int(input_image_id)]
    
    # Save the image to a bytes buffer
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Return the image
    return send_file(img_buffer, mimetype='image/png')

def main():
    # Context.t2i_model = Text2ImageModel()
    if Context.t2i_model is None:
        # Context.t2i_model = Text2ImageModel()
        Context.t2i_model = FluxInference()
    
    # Serve flask app
    # app.run(port=5001, debug=False)
    socketio.run(app, host='127.0.0.1', port=5001, debug=False, log_output=True)

if __name__ == "__main__":
    main()