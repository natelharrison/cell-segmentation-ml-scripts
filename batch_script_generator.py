import argparse
import subprocess
import json
import os
from datetime import datetime
from pathlib import Path
from shutil import rmtree

# Current time for default naming
now = datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")

# Setting up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, required=True)  # Ensure this is provided
parser.add_argument('--save_name', type=str, default=date_string)
parser.add_argument('--skip_images', type=int, default=0, help="Number of images to skip from the beginning")
args = parser.parse_args()

# Kwargs list for the cellpose model parameters (if needed)
kwargs_list = [
    {"diameter": 60, "do_3D": True, "normalize": True}
]

# Model list (if you need to iterate over multiple models, else just use one)
# model_path_str = "/global/home/users/natelharrison/cellpose_residual_on_style_on_concatenation_off_128_64_overlap_2023_08_03_01_23_59.133837"
model_path_str = 'cyto3_restore'
model_path = Path(model_path_str)

# The batch script that will be written to each .sh file
batch_script = """#!/bin/sh
#SBATCH --qos=abc_normal
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx
#SBATCH --account=co_abc
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --ntasks=8
#SBATCH --mem=500G
#SBATCH --output={log_path}
#SBATCH --export=ALL

. {user_dir}/anaconda3/etc/profile.d/conda.sh
conda activate cellpose

python cellpose_run.py --image_path {image_path} --model {model_path} --save_name {save_name} --kwargs '{kwargs_str}' """

# Function to make directories
def make_dir(dir_path: Path, remove_dir=True):
    if dir_path.exists() and remove_dir is True:
        rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

# Main function
def main():
    # Getting the image folder path
    image_folder = Path(args.image_folder)
    # Fetching all images, skipping first 'n' images
    image_files = list(image_folder.glob('*'))[args.skip_images:]  # Adjust the glob pattern as needed to match image types

    # Getting kwargs as string
    kwargs_str = json.dumps(kwargs_list[0])

    # User's home directory
    user_dir = Path.home()
    # Batch directory name
    save_name = args.save_name
    # Creating a batch directory within the user's home directory
    script_batch_dir = user_dir / 'cellpose_run' / save_name
    make_dir(script_batch_dir)

    # Looping over images and creating a script for each
    for j, image_path in enumerate(image_files, start=args.skip_images):
        log_path = script_batch_dir / f"log_{j}.log"
        formatted_batch_script = batch_script.format(
            log_path=log_path,
            user_dir=user_dir,
            image_path=image_path,
            model_path=model_path,
            save_name=f"{save_name}_img_{j}",
            kwargs_str=kwargs_str
        )
        script_save_path = script_batch_dir / f"script_{j}.sh"
        script_save_path.write_text(formatted_batch_script)
        subprocess.run(['sbatch', script_save_path.as_posix()])

if __name__ == '__main__':
    main()
