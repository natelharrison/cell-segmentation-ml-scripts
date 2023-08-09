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
parser.add_argument('--image_path', type=str, default=None)
parser.add_argument('--batch_name', type=str, default=date_string)
args = parser.parse_args()

kwargs_list = [
    # {"diameter": 0, "do_3D": True, "resample": True, "min_size": 5000, "augment": True},
    # {"diameter": 0, "do_3D": True, "resample": True, "min_size": 5000, "augment": False},
    # {"diameter": 0, "do_3D": True, "resample": False, "min_size": 5000, "augment": True},
    {"diameter": 0, "do_3D": True, "resample": False, "min_size": 5000, "augment": False},

    # {"diameter": 30, "do_3D": True, "resample": True, "min_size": 5000, "augment": True},
    # {"diameter": 30, "do_3D": True, "resample": True, "min_size": 5000, "augment": False},
    # {"diameter": 30, "do_3D": True, "resample": False, "min_size": 5000, "augment": True},
    {"diameter": 30, "do_3D": True, "resample": False, "min_size": 5000, "augment": False},

    # {"diameter": 60, "do_3D": True, "resample": True, "min_size": 5000, "augment": True},
    # {"diameter": 60, "do_3D": True, "resample": True, "min_size": 5000, "augment": False},
    # {"diameter": 60, "do_3D": True, "resample": False, "min_size": 5000, "augment": True},
    {"diameter": 60, "do_3D": True, "resample": False, "min_size": 5000, "augment": False}
]

model_list = [
    # 64_combined epoch 200
    ("/clusterfs/fiona/segmentation_curation/training_data/combined_dataset/64_no_overlap/models/"
     "cellpose_residual_on_style_on_concatenation_off_64_no_overlap_2023_08_03_16_09_17.755759"),
    # 128_combined epoch 400
    ("/clusterfs/fiona/segmentation_curation/training_data/combined_dataset/128_64_overlap/models/"
     "cellpose_residual_on_style_on_concatenation_off_128_64_overlap_2023_08_03_01_23_59.133837"),
    # 64_default
    ("/clusterfs/fiona/segmentation_curation/cellpose_training/trained_models/training_trial_2/"
     "cellpose_residual_on_style_on_concatenation_off_cropping_output_2023_07_01_18_46_14.616214")
]

batch_script = """#!/bin/sh
#SBATCH --qos=abc_normal
#SBATCH --gres=gpu:1
#SBATCH --partition=abc
#SBATCH --account=co_abc
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --ntasks=5
#SBATCH --mem=125G
#SBATCH --output={log_path}
#SBATCH --export=ALL

### Run your command
. {user_dir}/anaconda3/etc/profile.d/conda.sh
conda activate cellpose

python cellpose_run.py --image_path {image_path} --model {model_path} --save_name {batch_name} --kwargs '{kwargs_str}' """

def make_dir(dir_path: Path, remove_dir=True):
    if dir_path.exists() and remove_dir is True:
        rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def main():
    image_path = Path(args.image_path)

    kwargs_str_list = [json.dumps(kwargs) for kwargs in kwargs_list]
    model_path_list = [Path(model) for model in model_list]

    user_dir = Path.home()
    batch_name = args.batch_name
    script_batch_dir = user_dir / 'cellpose_run' / batch_name
    make_dir(script_batch_dir)

    for model_path in model_path_list:
        for i, kwargs_str in enumerate(kwargs_str_list):
            script_save_dir = script_batch_dir / model_path.stem
            make_dir(script_save_dir, remove_dir=False)

            log_path = (script_save_dir / f"batch_{i}").as_posix()

            formatted_batch_script = batch_script.format(
                log_path = log_path,
                user_dir = user_dir,
                image_path = image_path,
                model_path = model_path,
                batch_name = batch_name,
                kwargs_str = kwargs_str
            )

            script_save_path = script_save_dir / f"batch_{i}.sh"
            script_save_path.write_text(formatted_batch_script)
            subprocess.run(['sbatch', script_save_path.as_posix()])

if __name__ == '__main__':
    main()