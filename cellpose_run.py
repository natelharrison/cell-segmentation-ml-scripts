import ast
import os
import argparse
from pathlib import Path
import logging
from cellpose import models, io
from cellpose.io import imread


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='')
parser.add_argument('--image_path', type=str, default='')
parser.add_argument('--model', type=str, default='cyto2')
parser.add_argument('--kwargs', type=str, default='{}')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)


# model_type='cyto' or 'nuclei' or 'cyto2'
def load_cellpose_modelpath(model_path: Path,
                            gpu: bool = True) -> models.CellposeModel:

    # load cellpose model
    print('Loading Cellpose Models from folder ...')
    model = models.CellposeModel(gpu=gpu, pretrained_model=model_path.as_posix())
    print(f"Loaded {model_path.stem}")

    return model

model_path = Path(args.model)
model = load_cellpose_modelpath(model_path)

# list of files
# PUT PATH TO YOUR FILES HERE!
file_path = Path(args.image_path)
file_name = file_path.stem
file = file_path.as_posix()

save_dir = file_path.parent / f"{model_path.stem}_predictions"
os.makedirs(save_dir, exist_ok=True)
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

kwargs = ast.literal_eval(args.kwargs)
kwargs = {"diameter": 60, "min_size": 4000, "do_3D":False}

masks, flows, styles = model.eval(image,
                                  do_3D=True,
                                  resample=True,
                                  progress=True,
                                  channels=channels,
                                  **kwargs)
io.save_masks(images=image,
              masks=masks,
              flows=flows,
              file_names=file_name,
              png=False,
              tif=True,
              channels=channels,
              savedir=save_dir)
