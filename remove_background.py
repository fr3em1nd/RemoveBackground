import os
#import sys
#import argparse
import numpy as np
from PIL import Image
import json

# pip install numpy
# 07.11.2021 current numpy-1.21.4
# pip install --upgrade Pillow
# 07.11.2021 current Pillow-8.4.0
# pip install flask
# 07.11.2021 current flask-2.0.2
# https://pytorch.org/
# conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# for testing of service
# pip install requests
# 07.11.2021 current requests-2.26.0

# based on https://www.youtube.com/watch?v=vieoHqt7pxo

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from Coding.modnet import MODNet

from flask import Flask, request, jsonify, send_file
import io
from io import BytesIO
# install flask
# install requests
# conda install pytorch==1.10.0 -ging nicht
# pip install --upgrade torch==1.10.0

# input parameter
#im_name = 'NarutoHokage.png'
#input_path = 'PicInput'
#output_path = 'PicOutput'
# Model has always static folder
model_source = './Model/modnet_photographic_portrait_matting.ckpt'
# if no ref-size from service set to 512
ref_size = 512 # define hyper-parameters
# if only greyscale should be returned
only_greyscale = 'Yes'

def process_image(im_name, im, ref_size, model_source):
    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # create MODNet and load the pre-trained ckpt
    #torch.cuda.empty_cache()
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet).cuda()
    modnet.load_state_dict(torch.load(model_source))
    modnet.eval()

    # read image
    #im = pillow_img
    #im = Image.open(os.path.join(input_path, im_name))

    # unify image channels to 3
    im = np.asarray(im)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # convert image to PyTorch tensor
    im = Image.fromarray(im)
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im.cuda(), True)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    matte_name = im_name.split('.')[0] + '.png'
    #Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(output_path, matte_name))
    return Image.fromarray(((matte * 255).astype('uint8')), mode='L')

def combined_display(image, matte):
  # calculate display resolution
  w, h = image.width, image.height
  rw, rh = 800, int(h * 800 / (3 * w))
  
  # obtain predicted foreground
  image = np.asarray(image)
  if len(image.shape) == 2:
    image = image[:, :, None]
  if image.shape[2] == 1:
    image = np.repeat(image, 3, axis=2)
  elif image.shape[2] == 4:
    image = image[:, :, 0:3]
  matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
  foreground = image * matte + np.full(image.shape, 255) * (1 - matte)
  foreground_Image = Image.fromarray(np.uint8(foreground))

  # combine image, foreground, and alpha into one line
  # combined = np.concatenate((image, foreground, matte * 255), axis=1)
  # combined = Image.fromarray(np.uint8(combined)).resize((rw, rh))
  return foreground_Image

# return PIL Image to web service
def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=100)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

#flask integration
app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})
        try:
            # get hyper parameter
            data = json.load(request.files['data'])
            ref_size = data['ref-size']
            # get image data
            image_bytes = file.read()
            pillow_img = Image.open(io.BytesIO(image_bytes))
            im_no_background = process_image(file.filename, pillow_img, ref_size, model_source)
            #combine images
            full_img_no_background = combined_display(pillow_img, im_no_background)
            # greyscale will have less data, but it needs to be processed in the frontend
            if data['greyscale'] == 'Yes':
                return serve_pil_image(im_no_background)
            else:
                return serve_pil_image(full_img_no_background)
        except Exception as e:
            return jsonify({"error": str(e)})
    if request.method == "GET":
        return "OK"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)


#im = Image.open(os.path.join(input_path, im_name))