import argparse
import subprocess
import json

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='')
parser.add_argument('--model', type=str, default='cyto2')
args = parser.parse_args()

# Define your list of kwargs here
kwargs_list = [
    {"diameter": 60, "min_size": 1000},
    # Add more kwargs dictionaries here as needed
]


dir = Path.cwd()
user_dir = dir.parent
image_path = Path(args.image)
model_path = Path(args.model)

for i, kwargs in enumerate(kwargs_list):
    # Convert kwargs dictionary to a string
    kwargs["savedir"] = (image_path/f"batch_{i}").as_posix()
    kwargs_str = json.dumps(kwargs)

    # Create the batch script
    batch_name = f'batch_{i}_{image_path.stem}'
    log_output = dir/'logs'/f'{batch_name}.log'


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

python cellpose_run.py --image_path {image_path} --add_model {model_path} --kwargs '{kwargs_str}' """

    # Save the batch script to a file
    with open(f'{batch_name}.sh', "w") as file:
        file.write(batch_script)

    # Run the batch script
    subprocess.run(["sbatch", f"{batch_name}.sh"])


