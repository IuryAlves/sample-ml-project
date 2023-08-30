import tensorflow as tf
import tensorflow_hub as hub
import bentoml
import logging

logger = logging.getLogger(__name__)


model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2")
])
model.build([None, 224, 224, 3])

saved_model = bentoml.keras.save_model("model", model)

logger.info(f"Model saved: {saved_model}")
