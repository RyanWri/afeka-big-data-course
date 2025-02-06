import torch
from pyspark import SparkContext
from src.fsrcnn import FSRCNN


def load_fsrcnn_model(upscale_factor, model_path):
    """
    Loads the FSRCNN model from a checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FSRCNN(upscale_factor=upscale_factor)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def broadcast_model(spark_context: SparkContext, upscale_factor: int, model_path: str):
    """
    Loads and broadcasts the FSRCNN model across Spark workers.
    """
    model = load_fsrcnn_model(upscale_factor, model_path)
    return spark_context.broadcast(model)


def run_inference(patch_vector, model_broadcast, patch_size):
    """
    Applies FSRCNN super-resolution to a grayscale patch.
    """
    model = model_broadcast.value  # Get the model from broadcasted variable
    patch_tensor = (
        torch.tensor(patch_vector, dtype=torch.float32).reshape(
            1, 1, patch_size, patch_size
        )
        / 255.0
    )

    with torch.no_grad():
        sr_patch_tensor = model(patch_tensor)

    sr_patch_array = (sr_patch_tensor.squeeze().numpy() * 255.0).astype("uint8")
    return sr_patch_array.flatten().tolist()
