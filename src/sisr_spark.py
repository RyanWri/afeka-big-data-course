from src.fsrcnn import FSRCNN
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        "/home/ran/Documents/afeka/big-data/models/fsrcnn_x2-T91-f791f07f.pth.tar",
        map_location=device,
        weights_only=True,
    )
    model = FSRCNN(upscale_factor=2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    local_image_path = "/home/ran/datasets/spark-picsum-images/001.jpg"
    original_image = Image.open(local_image_path).convert("L")
    transform = transforms.ToTensor()  # Convert to PyTorch tensor
    image_tensor = transform(original_image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output_tensor = model(image_tensor)
    output_image = (
        output_tensor.squeeze().clamp(0, 1).numpy()
    )  # Remove batch dim and normalize
    output_image = (output_image * 255).astype(np.uint8)  # Convert to uint8
    result = Image.fromarray(output_image)
    result.save("fsrcnn_result.jpg")
    print("Super-resolved image using fsrcnn saved")
