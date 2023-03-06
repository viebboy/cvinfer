"""
processor.py: sample preprocessing and postprocessing script for deployment
---------------------------------------------------------------------------


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

from cvinfer.common import (
    Frame,
    Point,
    BoundingBox,
)



def preprocess(frame: Frame, config:dict):
    """
    The signature of the preprocessing function always have 2 arguments
    The 1st argument is always a Frame object or a list of Frame objects
    The 2nd argument is always a dictionary that contains configuration value
    """

    # --------------- ALL IMPORTS GO HERE -------------------------------------
    # -------------------------------------------------------------------------

    import numpy as np

    # --------------- END OF IMPORTS ------------------------------------------
    # -------------------------------------------------------------------------

    # preprocessing for yolox model is very simple
    # we only need to resize (keep aspect ratio, pad constant is 114) to a target size
    width = config['width']
    height = config['height']

    # because input is a Frame, we could leverage the resize() method to perform resizing
    # before resizing, we need to keep track of original size
    metadata = {
        'height_before': frame.height(),
        'width_before': frame.width(),
    }

    # note that resizing with aspect ratio kept will return 2 values
    frame, ratio = frame.resize(
        new_width=width,
        new_height=height,
        keep_ratio=True,
        pad_constant=114,
    )

    # bookkeeping this ratio for postprocessing
    metadata['resize_ratio'] = ratio
    metadata['height_after'] = frame.height()
    metadata['width_after'] = frame.width()

    # then obtain the numpy array
    # calling frame.data() returns a numpy array of type uint8
    # we need to
    output = np.ascontiguousarray(frame.data(), dtype=np.float32)

    # pytorch model is channel first so need to rearrange
    output = np.expand_dims(np.transpose(output, (2, 0, 1)), axis=0)

    # the 2nd variable returned by preprocessing() is used for bookkeeping
    # information that might be needed during postprocessing, such as original
    # image size and resize ratio
    return output, metadata

def postprocess(model_output, metadata, config):
    """
    The signature of the postprocessing function always have 3 arguments
    The 1st argument is the output obtained from running the ONNX model
    The 2nd argument is the metadata (2nd variable) returned by the preprocessing function
    The 3rd argument is always a dictionary that contains postprocessing configurations

    The output of postprocessing depends on the specific application.
    The guideline is that output should be represented in datastructures specified in common

    For this particular example, we want to return a list of BoundingBox objects
    """

    # --------------- ALL IMPORTS GO HERE -------------------------------------
    # all imports related to this function should be imported locally
    # -------------------------------------------------------------------------

    import numpy as np

    # --------------- END OF IMPORTS ------------------------------------------
    # -------------------------------------------------------------------------


    # -------------- ALL SUB-LOGICS GO HERE -----------------------------------
    # all user-created functions should be defined inside
    # -------------------------------------------------------------------------

    # in this example, we need to use non-maximum suppression
    # these logics taken from YOLOX:
    # https://github.com/Megvii-BaseDetection/YOLOX/blob/5c110a25596ad8385e955ac4fc992ba040043fe6/yolox/utils/demo_utils.py
    def nms(boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""

        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None

        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets


    # -------------- END OF SUB-LOGICS ----------------------------------------
    # -------------------------------------------------------------------------

    # output of ONNX model is always a list
    # in our case, yolox only returns 1 output
    outputs = model_output[0].squeeze(0) # this should be a numpy array

    # the original logic for postprocessing can be found from here
    # https://github.com/Megvii-BaseDetection/YOLOX/blob/5c110a25596ad8385e955ac4fc992ba040043fe6/yolox/utils/demo_utils.py
    # copy directly from the above logic

    # img_size is the size after resizing
    img_size = (metadata['height_after'], metadata['width_after'])

    grids = []
    expanded_strides = []

    """
    # in original logic, strides is determined by p6 argument
    # we now access it from config dict
    # basically the strides's value depends on the FPN config
    # default to [8, 16, 32, 64]
    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]
    """
    strides = config['strides']

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)

    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    # now the outputs contain both box predictions and confidence scores
    # logic taken from line 75 to line 88 in here:
    # https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/onnx_inference.py

    # first 4 dimensions are coordinates
    boxes = outputs[:, :4]
    # the 5th and 6th are objectness and class probability
    # multiplying them gives confidence score
    scores = outputs[:, 4:5] * outputs[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.

    # now convert back to coordinates in the original image using the resize
    # ratio
    boxes_xyxy /= metadata['resize_ratio']

    dets = multiclass_nms_class_agnostic(
        boxes_xyxy,
        scores,
        nms_thr=config['nonmaximum_suppression_threshold'],
        score_thr=config['confidence_threshold'],
    )

    original_height = metadata['height_before']
    original_width = metadata['width_before']

    outputs = []
    if dets is not None:
        # if not None, it should be a numpy array
        # first 4 dimensions are coordinates in x0, y0, x1, y1 order
        # the 5th dimension is the confidence score
        # the 6th dimension is the class index
        for item in dets:
            x0 = max(0, int(item[0]))
            y0 = max(0, int(item[1]))
            x1 = min(original_width, int(item[2]))
            y1 = min(original_height, int(item[3]))
            top_left = Point(x0, y0)
            bottom_right = Point(x1, y1)
            confidence = min(max(0.0, item[4]), 1.0)
            object_label = config['class_names'][int(item[5])]
            box = BoundingBox(
                top_left=top_left,
                bottom_right=bottom_right,
                confidence=confidence,
                label=object_label,
            )
            outputs.append(box)

    return outputs
