"""
detect_object.py: demo script on how to use the yolox artifacts
---------------------------------------------------------------


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
    OnnxModel,
)


if __name__ == '__main__':
    # data is the directory that is ignored in git
    output_dir = './data/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    onnx_local_path = os.path.join(output_dir, 'yolox_x.onnx')

    onnx_remote_path = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.onnx'
    if not os.path.exists(onnx_path):
        urllib.request.urlretrieve(onnx_remote_path, onnx_local_path)

    # use the artifacts via cvinfer.common.OnnxModel
    onnx_model = OnnxModel(
        onnx_path=onnx_local_path,
        processor_path='./processor.py',
        configuration_path='./configuration.json',
        execution_provider='CPUExecutionProvider',
    )

    # this model takes a frame as input
    frame = Frame('./dog_and_cat.jpeg')
    # calling this model on the frame will return a list of
    # cvinfer.common.BoundingBox instances
    bounding_boxes = onnx_model(frame)
    for box in bounding_boxes:
        # use draw_bounding_box() method to draw the bbox on the frame
        frame.draw_bounding_box(box)

    # save the result to
    frame.save('./data/detected_dog_and_cat.jpeg')
    logger.info('complete saving the output image to ./data/detected_dog_and_cat.jpeg')
