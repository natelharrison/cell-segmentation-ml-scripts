import argparse
import subprocess
import json
import os

from datetime import datetime
from pathlib import Path
from shutil import rmtree

now = datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='')
parser.add_argument('--model', type=str, default='')
parser.add_argument('--name', type=str, default=date_string)
args = parser.parse_args()

# Define your list of kwargs here
# 3D settings
kwargs_list = [
    {"diameter": 0, "do_3D": True, "resample": False, "mask_threshold": 0, "min_size": 5000},
    {"diameter": 0, "do_3D": True, "resample": False, "mask_threshold": 2, "min_size": 5000},
    {"diameter": 0, "do_3D": True, "resample": False, "mask_threshold": 4, "min_size": 5000},
    {"diameter": 30, "do_3D": True, "resample": True, "mask_threshold": 0, "min_size": 5000},
    {"diameter": 30, "do_3D": True, "resample": False, "mask_threshold": 0, "min_size": 5000},
    {"diameter": 30, "do_3D": True, "resample": False, "mask_threshold": 2, "min_size": 5000},
    {"diameter": 30, "do_3D": True, "resample": False, "mask_threshold": 4, "min_size": 5000},
    {"diameter": 60, "do_3D": True, "resample": False, "mask_threshold": 0, "min_size": 5000},
    {"diameter": 60, "do_3D": True, "resample": False, "mask_threshold": 2, "min_size": 5000},
    {"diameter": 60, "do_3D": True, "resample": False, "mask_threshold": 4, "min_size": 5000},
]


model_list = [
    #128_zyx_scratch
    # "/clusterfs/fiona/segmentation_curation/training_data/rotated_cropped_data/128_all_planes/models/cellpose_residual_on_style_on_concatenation_off_128_all_planes_2023_07_26_17_13_05.002100",
    #128_zyx_cyto2
    # "/clusterfs/fiona/segmentation_curation/training_data/rotated_cropped_data/128_all_planes/models/cellpose_residual_on_style_on_concatenation_off_128_all_planes_2023_07_24_23_31_57.719699",
    #64_default
    # "/clusterfs/fiona/segmentation_curation/training_data/rotated_cropped_data/cropping_output/models/cellpose_residual_on_style_on_concatenation_off_cropping_output_2023_07_01_18_46_14.616214",
    #128_zyx_cyto2_round2
    "/clusterfs/fiona/segmentation_curation/training_data/rotated_cropped_data/128_nonoverlap/models/cellpose_residual_on_style_on_concatenation_off_128_nonoverlap_2023_07_28_04_39_14.117439",
    #128_zyx_cyto2_additional
    # "/clusterfs/fiona/segmentation_curation/training_data/rotated_cropped_data/128_nonoverlap/models/cellpose_residual_on_style_on_concatenation_off_128_nonoverlap_2023_07_28_20_48_11.645318"
]


dir = Path.cwd()
user_dir = dir.parent
save_dir = user_dir/'cellpose_run'/args.name
log_dir = save_dir

if save_dir.exists():
    rmtree(save_dir)
os.mkdir(save_dir)

image_path = Path(args.image)
if args.model:
    model_list = [Path(args.model)]
else:
    model_list = [Path(model) for model in model_list]


for i, kwargs in enumerate(kwargs_list):
    # Convert kwargs dictionary to a string
    kwargs_str = json.dumps(kwargs)

    # Create the batch script
    batch_name = f'batch_{i}_{image_path.stem}'
    log_output = log_dir/f'{batch_name}.log'

    model_path = model_list[i % len(model_list)]

    # Rest of your script...
    batch_script = f"""#!/bin/sh
#SBATCH --qos=abc_normal
#SBATCH --gres=gpu:1
#SBATCH --partition=abc
#SBATCH --account=co_abc
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --ntasks=5
#SBATCH --mem=125G
#SBATCH --output={log_output}
#SBATCH --export=ALL

### Run your command
. {user_dir}/anaconda3/etc/profile.d/conda.sh
conda activate cellpose

python cellpose_run.py --image_path {image_path} --model {model_path} --save_name {batch_name} --kwargs '{kwargs_str}' """

    # Save the batch script to a file
    with open(save_dir / f'{batch_name}.sh', "w") as file:
        file.write(batch_script)

    # Run the batch script
    subprocess.run(["sbatch", (save_dir / f"{batch_name}.sh").as_posix()])



