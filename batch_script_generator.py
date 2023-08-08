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

kwargs_list = [
    {"diameter": 0, "do_3D": True, "resample": True, "min_size": 5000, "augment": True},
    {"diameter": 0, "do_3D": True, "resample": True, "min_size": 5000, "augment": False},
    {"diameter": 0, "do_3D": True, "resample": False, "min_size": 5000, "augment": True},
    {"diameter": 0, "do_3D": True, "resample": False, "min_size": 5000, "augment": False},

    {"diameter": 30, "do_3D": True, "resample": True, "min_size": 5000, "augment": True},
    {"diameter": 30, "do_3D": True, "resample": True, "min_size": 5000, "augment": False},
    {"diameter": 30, "do_3D": True, "resample": False, "min_size": 5000, "augment": True},
    {"diameter": 30, "do_3D": True, "resample": False, "min_size": 5000, "augment": False},

    {"diameter": 60, "do_3D": True, "resample": True, "min_size": 5000, "augment": True},
    {"diameter": 60, "do_3D": True, "resample": True, "min_size": 5000, "augment": False},
    {"diameter": 60, "do_3D": True, "resample": False, "min_size": 5000, "augment": True},
    {"diameter": 60, "do_3D": True, "resample": False, "min_size": 5000, "augment": False}
]

model_list = [
    # 64_combined epoch 200
    "/clusterfs/fiona/segmentation_curation/training_data/combined_dataset/64_no_overlap/models/cellpose_residual_on_style_on_concatenation_off_64_no_overlap_2023_08_03_16_09_17.755759"
    # 128_combined epoch 400
    "/clusterfs/fiona/segmentation_curation/training_data/combined_dataset/128_64_overlap/models/cellpose_residual_on_style_on_concatenation_off_128_64_overlap_2023_08_03_01_23_59.133837"
    # 64_default
    "/clusterfs/fiona/segmentation_curation/cellpose_training/trained_models/training_trial_2/cellpose_residual_on_style_on_concatenation_off_cropping_output_2023_07_01_18_46_14.616214"
]

dir = Path.cwd()
user_dir = dir.parent
save_dir = user_dir / 'cellpose_run' / args.name
log_dir = save_dir

if save_dir.exists():
    rmtree(save_dir)
os.mkdir(save_dir)

image_path = Path(args.image)
if args.model:
    model_list = [Path(args.model)]
else:
    model_list = [Path(model) for model in model_list]

for model in model_list:
    for i, kwargs in enumerate(kwargs_list):
        # Convert kwargs dictionary to a string
        kwargs_str = json.dumps(kwargs)

        # Create the batch script
        batch_name = f'batch_{i}_{image_path.stem}'
        log_output = log_dir / f'{batch_name}.log'

        model_path = model

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
