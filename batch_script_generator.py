import argparse
import subprocess
import os
from datetime import datetime
from pathlib import Path
from shutil import rmtree

# Current time for default naming
now = datetime.now()
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")

# Setting up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--denoise', action='store_true', default=False)
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=date_string, help="Changes default output dir")
parser.add_argument('--start_image', type=int, default=0, help="Index of the first image to process (0-indexed)")
parser.add_argument('--end_image', type=int, default=None, help="Index of the last image to process (0-indexed, inclusive)")
args = parser.parse_args()

# Batch script setup
batch_script = """#!/bin/sh
#SBATCH --qos=abc_normal
#SBATCH --gres=gpu:1
#SBATCH --partition=abc_a100
#SBATCH --account=co_abc
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --ntasks=4
#SBATCH --mem=125G
#SBATCH --output={log_path}
#SBATCH --export=ALL

. {user_dir}/anaconda3/etc/profile.d/conda.sh
conda activate cellpose

python cellpose_run.py --image {image_path} --model {model_path} --output_dir {output_dir} {denoise_option}"""

def make_dir(dir_path: Path, remove_dir=True):
    if dir_path.exists() and remove_dir:
        rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def main():
    image_folder = Path(args.image_folder)
    all_image_files = list(image_folder.glob('*.tif'))  # assuming TIFF format
    image_files = all_image_files[
        args.start_image:args.end_image + 1] if args.end_image is not None else all_image_files[args.start_image:]

    user_dir = Path.home()
    script_batch_dir = user_dir / 'cellpose_run' / args.output_dir
    make_dir(script_batch_dir)

    for j, image_path in enumerate(image_files, start=args.start_image):
        log_path = script_batch_dir / f"log_{j}.log"
        denoise_option = "--denoise" if args.denoise else ""
        formatted_batch_script = batch_script.format(
            log_path=log_path,
            user_dir=user_dir,
            image_path=image_path,
            model_path=args.model,
            output_dir=args.output_dir,
            denoise_option=denoise_option
        )
        script_save_path = script_batch_dir / f"script_{j}.sh"
        script_save_path.write_text(formatted_batch_script)
        result = subprocess.run(['sbatch', script_save_path.as_posix()], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to submit job for {image_path}: {result.stderr}")
        else:
            print(f"Submitted {script_save_path} to SLURM: {result.stdout}")

if __name__ == '__main__':
    main()
