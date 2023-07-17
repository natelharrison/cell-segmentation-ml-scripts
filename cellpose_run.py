import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from cellpose.io import imread


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='')
parser.add_argument('--image_dir', type=str, default='')
parser.add_argument('--model', type=str, default='cyto2')
args = parser.parse_args()

# model_type='cyto' or 'nuclei' or 'cyto2'

model_arg = args.model
model = models.Cellpose(gpu=True, model_type=model_arg)

# list of files
# PUT PATH TO YOUR FILES HERE!
file_path = Path(args.image_path)
file_name = file_path.stem
file = file_path.as_posix()

save_dir = file_path.parent / "predictions"
os.mkdir(save_dir)
save_dir = save_dir.as_posix()

image = imread(file)

# define CHANNELS to run segmentation on
# grayscale=0, R=1, G=2, B=3
# channels = [cytoplasm, nucleus]
# if NUCLEUS channel does not exist, set the second channel to 0
channels = [[0,0]]
# IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
# channels = [0,0] # IF YOU HAVE GRAYSCALE
# channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
# channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

# if diameter is set to None, the size of the cells is estimated on a per-image basis
# you can set the average cell `diameter` in pixels yourself (recommended)
# diameter can be a list or a single number for all images

masks, flows, styles, diams = model.eval(image,
                                         do_3D=True,
                                         progress=True,
                                         min_size=1000,
                                         channels=channels)
io.save_masks(images=image,
              masks=masks,
              flows=flows,
              file_names=file_name,
              png=False,
              tif=True,
              channels=channels,
              savedir=save_dir)