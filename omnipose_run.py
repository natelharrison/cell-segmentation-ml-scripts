import os
import torch

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from cellpose_omni import io
from cellpose_omni import models
from cellpose_omni import models, core
from omnipose.utils import normalize99
from cellpose_omni import io, transforms


# This checks to see if you have set up your GPU properly.
# CPU performance is a lot slower, but not a problem if you
# are only processing a few images.
use_GPU = core.use_gpu()
print('>>> GPU activated? %d'%use_GPU)

# for plotting
mpl.rcParams['figure.dpi'] = 300
plt.style.use('dark_background')


basedir = os.path.join(Path.cwd().parent,'test_files_3D')
files = io.get_image_files(basedir)

imgs = [io.imread(f) for f in files]

# print some info about the images.
for i in imgs:
    print('Original image shape:',i.shape)
    print('data type:',i.dtype)
    print('data range:', i.min(),i.max())
nimg = len(imgs)
print('number of images:',nimg)

model_name = 'plant_omni'

dim = 3
nclasses = 3 # flow + dist + boundary
nchan = 1
omni = 1
rescale = False
diam_mean = 0
use_GPU = 0 # Most people do not have enough VRAM to run on GPU... 24GB not enough for this image, need nearly 48GB
model = models.CellposeModel(gpu=use_GPU, model_type=model_name, net_avg=False,
                             diam_mean=diam_mean, nclasses=nclasses, dim=dim, nchan=nchan)

torch.cuda.empty_cache()
mask_threshold = -5 #usually this is -1
flow_threshold = 0.
diam_threshold = 12
net_avg = False
cluster = False
verbose = 1
tile = True
chans = None
compute_masks = 1
resample=False
rescale=None
omni=True
flow_factor = 10 # multiple to increase flow magnitude, useful in 3D
transparency = True

nimg = len(imgs)
masks_om, flows_om = [[]]*nimg,[[]]*nimg

# splitting the images into batches helps manage VRAM use so that memory can get properly released
# here we have just one image, but most people will have several to process
for k in range(nimg):
    masks_om[k], flows_om[k], _ = model.eval(imgs[k],
                                             channels=chans,
                                             rescale=rescale,
                                             mask_threshold=mask_threshold,
                                             net_avg=net_avg,
                                             transparency=transparency,
                                             flow_threshold=flow_threshold,
                                             omni=omni,
                                             resample=resample,
                                             verbose=verbose,
                                             diam_threshold=diam_threshold,
                                             cluster=cluster,
                                             tile=tile,
                                             compute_masks=compute_masks,
                                             flow_factor=flow_factor) 