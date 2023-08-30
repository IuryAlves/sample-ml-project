from typing import Dict, Tuple

import logging
import tensorflow as tf
import os
import cv2

from google.cloud import storage

import os

logger = logging.getLogger(__name__)


LOCAL_DIRECTORY = 'artifacts'

def download_object(gcs_path, destination_file_name):
    """Downloads an object from the given GCS path."""
    # Initialize the client
    storage_client = storage.Client()

    # Get the bucket and blob name from the GCS path
    bucket_name = gcs_path.split("/")[-2]
    blob_name = gcs_path.split("/")[-1]

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(blob_name)

    logging.info("Download the photo from GCS to local storage.")
    # Download the blob's content into a local file
    blob.download_to_filename(destination_file_name)

def preprocessing(
    path: str,
    image_shape: Tuple[int, int, int],
    image_scale: float,
) -> Tuple[tf.Tensor, str]:
    try:
        os.makedirs(os.makedirs(LOCAL_DIRECTORY))
        logging.info(f"Directory '{LOCAL_DIRECTORY}' created successfully.")
    except OSError as e:
        logging.info(f"Error creating directory '{LOCAL_DIRECTORY}': {e}")
    destination_file_name=os.path.join(os.path.dirname(__file__), LOCAL_DIRECTORY, path.split("/")[-1])
    download_object(
        path,
        destination_file_name=destination_file_name
    )
    image = tf.io.read_file(destination_file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    gray_image = tf.image.rgb_to_grayscale(image)
    is_blur, blur_coeff = is_blurry(image)
    if tf.reduce_mean(gray_image) <= 30:
        logging.info("Image is too dark.")
        return "PhotoDark", dict(zip(["PhotoDark", "PhotoBlurry"], [float(tf.reduce_mean(gray_image).numpy()), float(round(blur_coeff, 4))]))
    if is_blur:
        logging.info("Image is too blurry.")
        return "PhotoBlurry", dict(zip(["PhotoDark", "PhotoBlurry"], [float(tf.reduce_mean(gray_image).numpy()), float(round(blur_coeff, 4))]))
    image = tf.image.resize(image, image_shape[:2], method="bilinear")
    image.set_shape(image_shape)
    return tf.cast(image, tf.float32) / image_scale, None

def is_blurry(image, threshold=100):
    image = image.numpy()
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

def is_blurry_(image, threshold=100):
    gray_image = tf.image.rgb_to_grayscale(image)
    gray_image = tf.cast(gray_image, tf.float32)  # Convert to float32
    gray_image = tf.expand_dims(gray_image, 0)
    sobel_edges = tf.image.sobel_edges(gray_image)
    sobel_magnitude = tf.sqrt(tf.reduce_sum(sobel_edges ** 2, axis=-1))
    laplacian_var = tf.math.reduce_std(sobel_magnitude)
    return laplacian_var < threshold, laplacian_var

def parse(
    path: tf.Tensor,
    label: tf.Tensor,
    image_shape: Tuple[int, int, int],
    image_scale: float,
) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_shape[:2], method="bilinear")
    image.set_shape(image_shape)
    image = tf.cast(image, tf.float32) / image_scale
    return dict(image=image), label