# Cell Segmentation ML Scripts

## Description
This repo is still under work and as such is rapidly changing and a little messy for the sake of development time. The main goal is to currently get the segmentations masks we need and then I will spend time cleaning up and optimizing scripts for speed and reliability. If you would like to use any of these scipts and they are not working submit an issue and I can help as needed.

Scripts useful for training Cellpose and Omnipose models and running predictions on using trained models. These are used for 3D microscopy images in TIF format, however can be adapted to be used on other structured data.

## Getting Started

### Dependencies
* Python 3.8 or higher
* Cellpose and/or Omnipose, please follow their instructions for installing each.
* Additionally Required Libraries: Ray, Scikit-Optimize, Simple-ITK, Dask/Dask-Cuda
* OS: Windows/Linux/MacOS

## Executing program

#### `preprocessing.py`
* **Description**: Preprocesses image data for machine learning tasks, supporting cropping and dataset splitting.
* **Arguments**:
  * `--dir` (str, absolute path): The directory containing input TIFF images.
  * `--test_size` (float): Proportion of the dataset to use as the test set.
  * `--crop_size` (list of int, separated by spaces): The crop size dimensions [z, y, x].
  * `--strides` (list of int, separated by spaces, optional): The strides for cropping in each dimension.
  * `--save_name` (str): Base name for output directories.
  * `--remove_label` (int): A label in the mask to be removed.
* **Usage**:
  ```bash
  python /absolute/path/to/preprocessing.py --dir /path/to/directory --test_size 0.2 --crop_size 1 64 64 --strides 1 32 32 --save_name processed --remove_label 0

#### `postprocessing.py`
* **Description**: Applies postprocessing to image data, including label filtering and tile stitching.
* **Arguments**:
  * `--dir` (str, absolute path): The directory containing input TIFF images.
* **Usage**:
  ```bash
  python /absolute/path/to/postprocessing.py --dir /path/to/directory

#### `omnipose_run.py`
* **Description**: Runs image segmentation using the OmniPose model.
* **Arguments**:
  * `--image` (str, absolute path): Path to the input image.
  * `--mask` (str, absolute path, optional): Path to the input mask.
  * `--model` (str, absolute path): Path to the OmniPose model.
  * `--save_name` (str): Name for the output.
* **Usage**:
  ```bash
  python /absolute/path/to/omnipose_run.py --image /path/to/image --mask /path/to/mask --model /path/to/model --save_name output

#### `cellpose_run.py`
* **Description**: Applies the CellPose model for image segmentation.
* **Arguments**:
  * `--dir` (str, absolute path): The directory for input/output.
  * `--image_path` (str, absolute path): Path to the input image.
  * `--model` (str, absolute path): Path to the CellPose model.
  * `--pretrained` (str, absolute path, optional): Path to a pretrained model.
  * `--chunks` (list of int, separated by spaces, optional): Chunk sizes for processing.
  * `--kwargs` (str, JSON format): Additional arguments for the CellPose model.
  * `--save_name` (str): Name for the output.
  * `--batch_num` (str, optional): Batch number.
  * `--split` (str, optional): Data split method.
* **Usage**:
  ```bash
  python /absolute/path/to/cellpose_run.py --dir /path/to/directory --image_path /path/to/image --model /path/to/model --pretrained /path/to/pretrained --chunks 500 500 --kwargs '{"diameter": 30, "do_3D": true}' --save_name output --batch_num 1 --split method

#### `clean_training_data.py`
* **Description**: Cleans training data, with options for background removal and visualization.
* **Arguments**:
  * `--image_path` (str, absolute path): Path to the input image.
  * `--mask_path` (str, absolute path): Path to the input mask.
  * `--num_chunks` (int, optional): Number of chunks for parallel processing.
  * `--object_store_memory` (int, optional): Memory for Ray's object store.
  * `--background` (int, optional): Specify to remove background if labeled.
  * `--visualize` (flag, optional): Enable visualization.
* **Usage**:
  ```bash
  python /absolute/path/to/clean_training_data.py --image_path /path/to/image --mask_path /path/to/mask --num_chunks 4 --object_store_memory 5000000000 --background 1 --visualize

#### `batch_script_generator.py`
* **Description**: Generates and executes batch scripts for image processing tasks.
* **Arguments**:
  * `--image_path` (str, absolute path): Path to the input image.
  * `--save_name` (str): Name for the output.
* **Usage**:
  ```bash
  python /absolute/path/to/batch_script_generator.py --image_path /path/to/image --save_name output

#### `omnipose_loss_visualization.ipynb`
* **Description**: Jupyter notebook for visualizing the loss of the OmniPose model.
* **How to run**: Open the notebook in Jupyter and run the cells.

