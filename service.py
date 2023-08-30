import bentoml
import numpy as np
import json
import tensorflow as tf
import logging

from bentoml.io import JSON
from src.data import preprocessing, parse

logger = logging.getLogger(__name__)
runner = bentoml.keras.get("model").to_runner()
svc = bentoml.Service("test_model", runners=[runner])

with open('configs/main.json', 'r') as j:
     config = json.load(j)

classify_duration = bentoml.metrics.Histogram(
        name="classify_duration",
        duration="Duration of Classify",
        labelnames=["model_version"])


@svc.api(input=JSON(), output=JSON())
def classify(meta_json: json) -> json:
    image, score = preprocessing(meta_json['photoref'], config["shape"], config["scale"])

    if not isinstance(image, tf.Tensor):
        return {
            "positiveLabels": image,
            "negativeLabels": [],
            "rawOutput": score,
            "modelInfo": {"modelVersion": config["model_version"], "modelTimestamp": config["model_timestamp"]}
        }

    prediction = runner.predict.run(np.array([image.numpy()]))[0]

    positive_labels, negative_labels = [], []

    for index, pred in enumerate(prediction):
        if pred >= config["threshold"]:
            positive_labels.append(config["labels"][index])
        else:
            negative_labels.append(config["labels"][index])

    if not positive_labels:
        positive_labels.append("ParkingNone")

    raw_output = [float(round(x, 4)) for x in prediction]

    response = {
        "positiveLabels": positive_labels,
        "negativeLabels": negative_labels,
        "rawOutput": dict(zip(config["labels"], raw_output)),
        "modelInfo": {"modelVersion": config["model_version"], "modelTimestamp": config["model_timestamp"]}
    }
    return response
