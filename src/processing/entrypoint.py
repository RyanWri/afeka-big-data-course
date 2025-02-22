import os
import yaml
from src.processing.mass_split import main as msmain
from src.processing.batch_inference import main as bmain
from src.processing.mass_reconstruct import main as mrmain

def create_dynamic_config():
    base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    config_file = f"{base_folder}/src/processing/config.yaml"
    if os.path.isfile(config_file):
        print("config.yaml file already exists.")
        return

    conf = {"dataset": {
                "path": base_folder,
                "low_resolution_dir": f"{base_folder}/LR",
                "patches_dir": f"{base_folder}/LR-Patches",
                "inference_result_dir": f"{base_folder}/SR-Patches",
                "reconstructed_images_dir": f"{base_folder}/SR",
                "original_image_height": 64,
                "original_image_width": 64
    }}
    conf.update({"processing": {
                    "model_path": "src/processing/fsrcnn_x2-T91-f791f07f.pth.tar",
                    "patch_size": 16,
                    "upscale_factor": 2
    }})

    yaml_str = yaml.dump(conf, sort_keys=False, default_flow_style=False)
    with open(config_file, "w") as file:
        file.write(yaml_str)
    print("config.yaml file created successfully.")


def main(spark, sc, sqlContext):
    """
    PIPELINE STEPS:
    1. Split images into patches
    2. Run inference on patches
    3. Reconstruct images
    """
    create_dynamic_config()

    msmain(sqlContext)
    bmain(sc, spark)
    mrmain(spark)

    print("All steps completed successfully.")
