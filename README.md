# Image Super-Resolution Pipeline with Apache Spark

This repository showcases an **end-to-end pipeline** for performing **image super-resolution** at scale using [FSRCNN](https://github.com/Lornatang/FSRCNN-PyTorch) and Apache Spark. The architecture is split into three main clusters—**Ingestion**, **Processing**, and **Notifications**—to handle data ingestion, distributed image processing, and final notifications or downstream consumption.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Ingestion Cluster](#ingestion-cluster)
- [Processing Cluster](#processing-cluster)
- [Notifications Cluster](#notifications-cluster)
- [FSRCNN Model](#fsrcnn-model)
- [Pipeline Steps](#pipeline-steps)
- [Prerequisites](#prerequisites)
- [Running the Pipeline](#running-the-pipeline)
- [Automated Entry Point](#automated-entry-point)
- [Contributing](#contributing)
- [License](#license)

---

## Architecture Overview

Below is a high-level view of the complete architecture, illustrating how images progress from ingestion through processing to final consumption.

![Complete Architecture](architecture/images/complete_architecture.png)

1. **Ingestion Cluster**  
   - Fetches image data (e.g., from an external API or user uploads).  
   - Publishes image references or metadata to Kafka.  
   - Stores images in an object storage system (e.g., S3) partitioned by ingestion date or image ID.

2. **Processing Cluster**  
   - Uses Apache Spark for distributed processing.  
   - Splits images into patches, applies super-resolution, and reconstructs final images.  
   - Writes processed images back to object storage.

3. **Notifications Cluster**  
   - Publishes an event via REST callback once images are processed.  
   - Notifies downstream consumers or microservices that the new, high-resolution images are available.
   - The notification service is an independent Flask app.
   - The notification service visualizes the processed images and displays the evaluations.

---

## Ingestion Cluster

![Ingestion Cluster](architecture/images/ingestion_cluster.png)

- **Producer Service**: Retrieves images from an external API (e.g., [Picsum](https://picsum.photos/) or any custom data source).  
- **Kafka Topics**: Act as a buffer and reliable transport mechanism for high volumes of image references.  
- **Consumer Service**: Reads messages from Kafka, downloads the images, and stores them in an object storage bucket (such as S3) for later processing.
- **Threading**: For this project we decided to use threading for the Kafka producer and consumer to allow concurrent processing and high throughput.

This decouples the image acquisition rate from the downstream processing speed.

---

## Processing Cluster

![Processing Cluster](architecture/images/processing_cluster.png)

**Prerequisite**: We are running pyspark through jupyter notebook and we assume that the spark context and sqlcontext are well defined,
in a real world scenario we will prefer to use spark session and have full control over the spark session itself. 
The processing cluster orchestrates the Spark ETL pipeline for super-resolution:

1. **Fetch Batch**: Reads raw images from the S3 input bucket (or local storage, for a proof-of-concept).
2. **Patch Extraction** (`mass_split.py`):  
   - Reads each image, converts it to grayscale (if desired), and normalizes pixel values to `[0, 1]`.  
   - Splits the image into patches of size `patch_size` × `patch_size`.  
   - Writes the patches as a Parquet dataset.
3. **Model Inference** (`batch_inference.py`):  
   - Reads the patches dataset.  
   - Broadcasts the FSRCNN model to all Spark executors.  
   - Runs super-resolution on each patch, converting the values back to `[0, 255]`.  
   - Writes the super-resolved patches to another Parquet dataset.
4. **Reconstruction** (`mass_reconstruct.py`):  
   - Groups super-resolved patches by `image_id` and stitches them together.  
   - Saves the reconstructed high-resolution images (e.g., PNG) to the output bucket.
5. **Write Final Output**: The final images are stored in the S3 output bucket (or local directory).

---

## Notifications Cluster

Once the super-resolution images are produced, the Notifications Cluster reads the images via a REST API and displays the images that are ready. This event will trigger an update for the visualized images.
The notifications service displays the last 10 images (can be changed via code) and the evaluated metrics.
The notifications service refreshes every 90 seconds (also modifiable).

---

## FSRCNN Model

We use **[FSRCNN (Fast Super-Resolution Convolutional Neural Network)](https://github.com/Lornatang/FSRCNN-PyTorch)**, a lightweight and efficient model for image super-resolution. Key benefits include:

- **Speed**: Faster inference compared to many other super-resolution architectures.
- **Quality**: Significant improvements in visual clarity and detail.
- **Simplicity**: Easy integration with PyTorch and PySpark for distributed inference.

---

## Pipeline Steps

1. **Ingestion**  
   - Pull images from an external API or user input.
   - Store them in S3 (or a local folder) for Spark consumption.
2. **Patch Extraction** (`mass_split.py`)  
   - Convert images to grayscale and normalize to `[0, 1]`.
   - Split images into patches.
   - Save patches as a Parquet dataset.
3. **Model Inference** (`batch_inference.py`)  
   - Read the Parquet dataset.
   - Apply FSRCNN super-resolution to each patch.
   - Rescale patch values to `[0, 255]` after inference.
   - Save the super-resolved patches as a Parquet dataset.
4. **Reconstruction** (`mass_reconstruct.py`)  
   - Group patches by `image_id` and reconstruct high-resolution images.
   - Save final images (PNG format) to the output bucket.
5. **Notification**  
   - Display the created super-resolution images alongside the high-resolution images including the evaluated metrics.

---

## Prerequisites

- **Python**: Version 3.9 (avoid newer versions for compatibility).
- **Apache Spark**: Tested on Spark 3.3.0.
- **PyTorch**: Required for FSRCNN.
- **PySpark**: For running Spark jobs in Python.
- **Pillow (PIL)**: For image manipulation.
- **Jupyter Notebook**: Currently we run pyspark through Jupyter Notebook and assume that the spark context and sqlcontext are automatically created.
- **Flask**: Must run as a separate thread, otherwise it is blocking.
- **Confluent Kafka**: Must be installed for the Kafka producer and consumer services.

Ensure that all required Python libraries are listed in your [requirements.txt](requirements.txt).

---

## Running the Pipeline

**Run everything from root project directory**
1. **Run Kafka Start Script** (recommended):
   The script starts the Zookeeper and Kafka services and creates a new topic "images-topic".
   You might need to change the paths in the script to your desired Kafka directory location.
   ```bash
   ./scripts/kafka_start.sh

2. **Configure the Paths in config.yaml** (Optional):<br>
   **Config is created dynamically in main.**<br>
   dataset.low_resolution_dir: Directory (or S3 path) with raw images.<br>
   dataset.patches_dir: Output location for extracted patches.<br>
   dataset.inference_result_dir: Output location for super-resolved patches.<br>
   dataset.reconstructed_images_dir: Final directory for high-resolution images.<br>
   processing.model_path: Path to your FSRCNN model checkpoint.<br>
   processing.patch_size: Patch size used for splitting and reconstruction.<br>
   processing.upscale_factor: The super-resolution scale factor (e.g., 2 or 4).


3. **Run the Jupyter notebook main.ipynb**: 
   The notebook will run the entire program, including ingestion cluster, processing cluster and the notification service.


4. **Stop The Kafka Services**:
   The script stops the Zookeeper and Kafka services.
   You might need to change the paths in the script to your desired Kafka directory location.<br>
   **Note that the topic will NOT be removed when the services are stopped.**
   ```bash
   ./scripts/kafka_stop.sh

## Contributing
Contributions and feedback are welcome! If you have suggestions for improvements, new features, or bug fixes, please open a pull request or create an issue.

### License
This project is provided under an open-source license. See LICENSE for details.

### Happy Super-Resolving!
Enjoy scaling your images with FSRCNN Apache Kafka and Spark in a fully distributed environment.