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
parser.add_argument('--image_folder', type=str, default=None)  # Changed from image_path to image_folder
parser.add_argument('--save_name', type=str, default=date_string)
args = parser.parse_args()

kwargs_list = [
    {"diameter": 60, "do_3D": True, "min_size": 2000, "normalize": True}
]

model_list = [
    "/global/home/users/natelharrison/cellpose_residual_on_style_on_concatenation_off_128_64_overlap_2023_08_03_01_23_59.133837"
]

batch_script = """#!/bin/sh
#SBATCH --qos=abc_normal
#SBATCH --gres=gpu:1
#SBATCH --partition=abc_a100
#SBATCH --account=co_abc
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --ntasks=8
#SBATCH --mem=250G
#SBATCH --output={log_path}
#SBATCH --export=ALL

. {user_dir}/anaconda3/etc/profile.d/conda.sh
conda activate cellpose

python cellpose_run.py --image_path {image_path} --model {model_path} --save_name {save_name} --batch_num {batch_num} --kwargs '{kwargs_str}' """

def make_dir(dir_path: Path, remove_dir=True):
    if dir_path.exists() and remove_dir is True:
        rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def main():
    image_folder = Path(args.image_folder)
    image_files = list(image_folder.glob('*'))[:100]  # Adjust the glob pattern as needed to match image types

    kwargs_str_list = [json.dumps(kwargs) for kwargs in kwargs_list]
    model_path_list = [Path(model) for model in model_list]

    user_dir = Path.home()
    save_name = args.save_name
    script_batch_dir = user_dir / 'cellpose_run' / save_name
    make_dir(script_batch_dir)

    for model_path in model_path_list:
        for i, kwargs_str in enumerate(kwargs_str_list):
            script_save_dir = script_batch_dir / model_path.stem / f"batch_{i}"
            make_dir(script_save_dir, remove_dir=False)

            for j, image_path in enumerate(image_files):
                log_path = script_save_dir / f"log_{j}.log"
                formatted_batch_script = batch_script.format(
                    log_path=log_path,
                    user_dir=user_dir,
                    image_path=image_path,
                    model_path=model_path,
                    save_name=f"{save_name}_img_{j}",
                    batch_num=i,
                    kwargs_str=kwargs_str
                )
                script_save_path = script_save_dir / f"script_{j}.sh"
                script_save_path.write_text(formatted_batch_script)
                subprocess.run(['sbatch', script_save_path.as_posix()])

if __name__ == '__main__':
    main()
