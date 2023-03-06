"""
converter.py: sample converter script to generate artifacts for deployment
--------------------------------------------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran
* Emails: viebboy@gmail.com
* Date: 2022-06-14
* Version: 0.0.1

License
-------
Apache License 2.0


"""

from __future__ import annotations
import os
from loguru import logger
import json
import urllib.request
import shutil

from cvinfer.common import (
    Frame,
    Point,
    BoundingBox,
    OnnxModel,
)


COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def write_artifacts(prefix: str):
    """
    This function is used to write the artifacts
    """

    # -------------- WRITE DEPENDENCIES ------------------------------
    DEPENDENCIES = [
        'numpy',
        'cvinfer',
    ]

    logger.warning('Dependencies are the following')
    logger.warning(f'{DEPENDENCIES}')

    with open(prefix + '.dependencies.txt', 'w') as fid:
        for item in DEPENDENCIES:
            fid.write(item + '\n')
    # -------------- END OF WRITING DEPENDENCIES --------------------

    # -------------- WRITE SAMPLE CONFIG ----------------------------
    CONFIG = {
        'preprocessing': {
            'width': 640,
            'height': 640,
        },
        'postprocessing': {
            'strides': [8, 16, 32],
            'nonmaximum_suppression_threshold': 0.5,
            'confidence_threshold': 0.5,
            'class_names': COCO_CLASSES,
        }
    }
    logger.warning('writing the following values as into sample configuration file')
    print(json.dumps(CONFIG, indent=2))

    config_file = prefix + '.configuration.json'
    with open(config_file, 'w') as fid:
        fid.write(json.dumps(CONFIG, indent=2))

    logger.info(f'complete writing sample configurations to {config_file}')
    # -------------- END OF WRITING SAMPLE CONFIG -------------------


    # -------------- CONVERT PYTORCH/TF/MXNET MODEL TO ONNX ---------
    # ---------------------------------------------------------------

    #TODO: implement conversion and verification here
    # we dont need to implement conversion for yolox because we can download
    # pre-exported models

    # -------------- END OF CONVERSION ------------------------------
    # ---------------------------------------------------------------



if __name__ == '__main__':
    # data is the directory that is ignored in git
    output_dir = './data/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists('./data/dog_and_cat.jpeg'):
        msg = (
            'Cannot find "dog_and_cat.jpeg" under "./data/". ',
            'Please download any random dog and cat image and put it under ./data/'
        )
        raise RuntimeError(msg)

    prefix = os.path.join(output_dir, 'yolox_x')

    onnx_remote_path = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.onnx'
    if not os.path.exists(prefix + '.onnx'):
        urllib.request.urlretrieve(onnx_remote_path, prefix + '.onnx')

    # write artifacts
    write_artifacts(prefix)

    # copy the processor.py file to the correct path
    # if the model is saved as yolox_x.onnx then the python file that contains
    # preprocess and postprocess functions should be named yolox_x.processor.py
    shutil.copy('processor.py', prefix + '.processor.py')

    # testing the artifacts
    onnx_model = OnnxModel(prefix + '.onnx')
    frame = Frame('./data/dog_and_cat.jpeg')
    bounding_boxes = onnx_model(frame)
    for box in bounding_boxes:
        frame.draw_bounding_box(box)
    frame.save('./data/detected_dog_and_cat.jpeg')
