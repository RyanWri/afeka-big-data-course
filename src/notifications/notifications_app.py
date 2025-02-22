import os
import sys
import threading
from sympy import false

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flask import Flask, render_template, jsonify, send_from_directory
from src.notifications.comparison import evaluate_image_quality  # Import your function



BASE_DIR = os.getcwd()
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# Define paths for images
ORIGINAL_IMAGES_DIR = f"{BASE_DIR}/HR"  # High-Resolution images folder
SUPER_RES_IMAGES_DIR = f"{BASE_DIR}/SR"  # Super-Resolved images folder
NUM_IMAGES = 10 # Number of last processed images to display

def get_latest_images(directory):
    """Get the last NUM_IMAGES files sorted by modification time."""
    try:
        files = sorted(
            os.listdir(directory),
            key=lambda x: os.path.getmtime(os.path.join(directory, x)),
            reverse=True
        )
        return files[:NUM_IMAGES]
    except Exception as e:
        print(f"Error retrieving images from {directory}: {e}")
        return []

def get_latest_image():
    """Fetches the latest image file name from the HR directory."""
    try:
        images = sorted(os.listdir(ORIGINAL_IMAGES_DIR),
                        key=lambda x: os.path.getmtime(os.path.join(ORIGINAL_IMAGES_DIR, x)), reverse=True)
        return images[0] if images else None
    except Exception as e:
        print(f"Error retrieving images: {e}")
        return None


@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template("index.html")

@app.route('/latest_results')
def latest_results():
    """API endpoint that returns the latest images and their metrics."""
    sr_images = get_latest_images(SUPER_RES_IMAGES_DIR)
    hr_images = [f.replace("SR", "HR") for f in sr_images]

    if not hr_images or not sr_images:
        return jsonify({"error": "No images found"}), 404

    results = []
    for hr, sr in zip(hr_images, sr_images):
        # Evaluate PSNR and SSIM using your function
        metrics = evaluate_image_quality(hr)
        results.append({
            "original_image": f"/HR/{hr}",
            "super_res_image": f"/SR/{sr}",
            "psnr": round(metrics["PSNR"], 2),
            "ssim": round(metrics["SSIM"], 4)
        })

    return jsonify(results)

@app.route('/HR/<filename>')
def serve_hr(filename):
    return send_from_directory(ORIGINAL_IMAGES_DIR, filename)

@app.route('/SR/<filename>')
def serve_sr(filename):
    return send_from_directory(SUPER_RES_IMAGES_DIR, filename)

def run_app():
    app.run(debug=false, use_reloader=false, port=9000)

def main():
    thread = threading.Thread(target=run_app)
    thread.daemon = True
    thread.start()
