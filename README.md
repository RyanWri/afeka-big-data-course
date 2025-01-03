# Image Super Resolution using Apache Spark

This project demonstrates a Proof of Concept (POC) for performing **image super-resolution** using Apache Spark. The goal is to upscale an image by splitting it into patches, applying super-resolution to each patch independently in a distributed fashion, and then reconstructing the image.

## Features
- Splits a single image into smaller patches for parallel processing.
- Applies a basic super-resolution method (bicubic interpolation) to each patch.
- Reconstructs the final high-resolution image from the processed patches.
- Utilizes Apache Spark for distributed computation, ensuring scalability and efficiency.

## Technologies Used
- **Python**: Programming language for implementation.
- **Apache Spark**: Distributed computing framework for processing image patches in parallel.
- **Pillow**: Image processing library for handling image manipulation tasks.
- **PySpark**: Python API for Apache Spark.

## Prerequisites
- Python 3.9 - 3.11 (do not use newer python version) 
- Apache Spark
- Required Python libraries: `pyspark`

## running our code
 - please create virtual environment
 - install requirments.txt
 - entry point is src/main.py