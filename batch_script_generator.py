import argparse
import subprocess
import json
import os

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='')
parser.add_argument('--model', type=str, default='cyto2')
args = parser.parse_args()

# Define your list of kwargs here
# 3D settings
kwargs_list = [
    #3D
    {"diameter": 0, "do_3D": True, "resample": True},
    {"diameter": 64, "do_3D": True, "resample": True},
    {"diameter": 58, "do_3D": True, "resample": True},
    {"diameter": 30, "do_3D": True, "resample": True},
    {"diameter": 0, "do_3D": True, "resample": False},
    {"diameter": 64, "do_3D": True, "resample": False},
    {"diameter": 58, "do_3D": True, "resample": False},
    {"diameter": 30, "do_3D": True, "resample": False},
    {"diameter": 0, "do_3D": True, "resample": True, "min_size": 4000},
    {"diameter": 64, "do_3D": True, "resample": True, "min_size": 4000},
    {"diameter": 58, "do_3D": True, "resample": True, "min_size": 4000},
    {"diameter": 30, "do_3D": True, "resample": True, "min_size": 4000},

    #2D
    # {"diameter": 0, "do_3D": False, "resample": True, "cellprob_threshold": 0.0, "flow_threshold": 0.4, "stitch_threshold": 0.0},
    # {"diameter": 64, "do_3D": False, "resample": True, "cellprob_threshold": 0.0, "flow_threshold": 0.4, "stitch_threshold": 0.0},
    # {"diameter": 58, "do_3D": False, "resample": True, "cellprob_threshold": 0.0, "flow_threshold": 0.4, "stitch_threshold": 0.0},
    # {"diameter": 30, "do_3D": False, "resample": True, "cellprob_threshold": 0.0, "flow_threshold": 0.4, "stitch_threshold": 0.0},
    # {"diameter": 0, "do_3D": False, "resample": False, "cellprob_threshold": 0.0, "flow_threshold": 0.4, "stitch_threshold": 0.0},
    # {"diameter": 64, "do_3D": False, "resample": False, "cellprob_threshold": 0.0, "flow_threshold": 0.4, "stitch_threshold": 0.0},
    # {"diameter": 58, "do_3D": False, "resample": False, "cellprob_threshold": 0.0, "flow_threshold": 0.4, "stitch_threshold": 0.0},
    # {"diameter": 30, "do_3D": False, "resample": False, "cellprob_threshold": 0.0, "flow_threshold": 0.4, "stitch_threshold": 0.0},
    # {"diameter": 0, "do_3D": False, "resample": True, "cellprob_threshold": 0.2, "flow_threshold": 0.6, "stitch_threshold": 0.2},
    # {"diameter": 64, "do_3D": False, "resample": True, "cellprob_threshold": 0.2, "flow_threshold": 0.6, "stitch_threshold": 0.2},
    # {"diameter": 58, "do_3D": False, "resample": True, "cellprob_threshold": 0.2, "flow_threshold": 0.6, "stitch_threshold": 0.2},
    # {"diameter": 30, "do_3D": False, "resample": True, "cellprob_threshold": 0.2, "flow_threshold": 0.6, "stitch_threshold": 0.2},
]


dir = Path.cwd()
user_dir = dir.parent
save_dir = user_dir/'cellpose_run'/'mass_batch'
log_dir = user_dir/'cellpose_run'/'logs'

image_path = Path(args.image)
model_path = Path(args.model)

for i, kwargs in enumerate(kwargs_list):
    # Convert kwargs dictionary to a string
    savedir_name = f"batch_{i}"
    kwargs_str = json.dumps(kwargs)

    # Create the batch script
    batch_name = f'batch_{i}_{image_path.stem}'
    log_output = log_dir/f'{batch_name}.log'


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

python cellpose_run.py --image_path {image_path} --model {model_path} --save_name {savedir_name} --kwargs '{kwargs_str}' """

    # Save the batch script to a file
    with open(save_dir / f'{batch_name}.sh', "w") as file:
        file.write(batch_script)

    # Run the batch script
    subprocess.run(["sbatch", (save_dir / f"{batch_name}.sh").as_posix()])



