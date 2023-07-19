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
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)


# model_type='cyto' or 'nuclei' or 'cyto2'
def load_model(
        model_path: Path,
        gpu: bool = True
) -> models.CellposeModel:
    """
    Load a Cellpose model from a given path.

    Parameters:
    model_path (Path): Path to the pretrained model.
    gpu (bool): If True, use GPU for model prediction. Default is True.

    Returns:
    models.CellposeModel: The loaded Cellpose model.
    """
    # load cellpose model
    print('Loading Cellpose Models from folder ...')
    model = models.CellposeModel(
        gpu=gpu,
        pretrained_model=model_path.as_posix()
    )
    print(f"Loaded {model_path.stem}")

    return model


def model_predictions(
        model,
        image,
        file_name,
        channels,
        save_dir,
        **kwargs
):
    """
    Run a Cellpose model on an image and save the results.

    Parameters:
    model (models.CellposeModel): The Cellpose model to use for prediction.
    image (np.ndarray): The image to run the model on.
    file_name (str): The name of the file to save the results to.
    channels (list): List of channels to run segmentation on.
    save_dir (str): The directory where the results will be saved.
    **kwargs: Additional parameters to pass to the model's eval method.

    Returns:
    None
    """
    masks, flows, styles = model.eval(
        image,
        progress=True,
        channels=channels,
        **kwargs
    )
    io.save_masks(
        images=image,
        masks=masks,
        flows=flows,
        file_names=file_name,
        png=False,
        tif=True,
        channels=channels,
        savedir=save_dir
    )


def main():
    """
    Main function to run the script. It loads the model, reads the image, and runs the model prediction.
    The results are saved in the specified directory.
    """
    model_path = Path(args.model)
    model = load_model(model_path)

    file_path = Path(args.image_path)
    file_name = file_path.stem
    file = file_path.as_posix()

    save_dir = file_path.parent / f"{model_path.stem}_predictions"
    os.makedirs(save_dir, exist_ok=True)
    save_dir = save_dir.as_posix()

    image = imread(file)

    channels = [[0,0]]  # define channels to run segmentation on

    model_predictions(
        model,
        image,
        file_name,
        channels,
        save_dir
    )

