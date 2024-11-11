"""
common.py: common interfaces
----------------------------


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
from typing import Union, Tuple, Any
import numpy as np
import os
import sys
from loguru import logger
import onnxruntime as ORT
import cv2
import time
from drawline import draw_rect
import json
import string
import random
import importlib.util
import dill


INTERPOLATIONS = {
    "cubic": cv2.INTER_CUBIC,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

MARKERS = [
    "circle",
    "cross",
    "tilted_cross",
    "star",
    "diamond",
    "square",
    "triangle_up",
    "triangle_down",
]


def load_module(module_file, attribute, module_name=None):
    if module_name is None:
        module_name = "".join(random.sample(string.ascii_letters, 10))

    try:
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error1:
        try:
            # try to append directory that contains module_file to path
            logger.debug(f"fails to import attribute {attribute} from module {module_file}")
            module_path = os.path.dirname(os.path.abspath(module_file))
            logger.debug(f"trying to append {module_path} to sys.path to fix this issue")
            sys.path.append(module_path)

            spec = importlib.util.spec_from_file_location(module_name, module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as error2:
            logger.error(
                f"ERROR when trying to load from {module_file} without appending parent dir"
            )
            logger.error(str(error1))
            logger.error(
                f"ERROR when trying to load from {module_file} WITH parent dir appended to sys.path"
            )
            logger.error(str(error2))
            raise error1

    if hasattr(module, attribute):
        return getattr(module, attribute)
    else:
        raise RuntimeError(
            f"Cannot find attribute {attribute} in the given module at {module_file}"
        )


class TimeMeasure:
    """
    convenient class to measure latency
    """

    def __init__(self, function_name, logger=logger):
        self.function_name = function_name
        self.logger = logger

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        stop = time.time()
        if stop - self.start < 1:
            duration = "{:.6f}".format(stop - self.start)
        else:
            duration = "{:.2f}".format(stop - self.start)
        self.logger.info(f"{self.function_name} took {duration} seconds")


class Color:
    """
    abstraction for color representation
    """

    def __init__(self, red=None, green=None, blue=None):
        if red is None:
            red = np.random.randint(0, 256)

        if green is None:
            green = np.random.randint(0, 256)

        if blue is None:
            blue = np.random.randint(0, 256)

        assert isinstance(red, int) and 0 <= red <= 255
        assert isinstance(green, int) and 0 <= green <= 255
        assert isinstance(blue, int) and 0 <= blue <= 255

        self._red = int(red)
        self._green = int(green)
        self._blue = int(blue)

    def rgb(self):
        return self._red, self._green, self._blue

    def bgr(self):
        return self._blue, self._green, self._red

    def __str__(self):
        return "(R={}, G={}, B={})".format(*self.rgb())


class OnnxModel:
    """
    base implementation of the onnx inference model
    """

    @classmethod
    def from_dir(
        asset_dir: str,
        relative_onnx_path: str = "model.onnx",
        relative_processor_path: str = "processor.py",
        relative_configuration_path: str = "configuration.json",
        execution_provider="CPUExecutionProvider",
        **kwargs: dict,
    ):
        return OnnxModel(
            onnx_path=os.path.join(asset_dir, relative_onnx_path),
            processor_path=os.path.join(asset_dir, relative_processor_path),
            configuration_path=os.path.join(asset_dir, relative_configuration_path),
            execution_provider=execution_provider,
            **kwargs,
        )

    def __init__(
        self,
        onnx_path,
        processor_path=None,
        configuration_path=None,
        execution_provider="CPUExecutionProvider",
        **kwargs: dict,
    ):
        # load preprocessing function
        if processor_path is None:
            processor_path = onnx_path.replace(".onnx", ".processor.py")

        assert os.path.exists(processor_path)
        self.preprocess_function = load_module(processor_path, "preprocess")
        self.postprocess_function = load_module(processor_path, "postprocess")

        # load onnx model from onnx_path
        avail_providers = ORT.get_available_providers()
        logger.info("all available ExecutionProviders are:")
        for idx, provider in enumerate(avail_providers):
            logger.info(f"\t {provider}")

        logger.info(f"trying to run with execution provider: {execution_provider}")
        if execution_provider == "CUDAExecutionProvider" and execution_provider in avail_providers:
            assert "device_id" in kwargs, "device_id must be provided for CUDAExecutionProvider"
            assert (
                "gpu_mem_limit" in kwargs
            ), "gpu_mem_limit must be provided for CUDAExecutionProvider"
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": kwargs["device_id"],
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": int(
                            kwargs["gpu_mem_limit"] * 1024 * 1024 * 1024
                        ),  # convert to bytes
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": False,
                        "cudnn_conv_use_max_workspace": "0",
                    },
                ),
            ]
        else:
            providers = [execution_provider]

        self.session = ORT.InferenceSession(onnx_path, providers=providers)

        self.input_names = [input_.name for input_ in self.session.get_inputs()]

        if configuration_path is None:
            configuration_path = onnx_path.replace(".onnx", ".configuration.json")

        if not os.path.exists(configuration_path):
            raise FileNotFoundError(configuration_path)

        with open(configuration_path, "r") as fid:
            # self.config is a dictionary
            self.config = json.loads(fid.read())

    @logger.catch
    def preprocess(self, inputs: Union[Frame, list[Frame]]):
        assert isinstance(inputs, Frame)
        return self.preprocess_function(inputs, self.config["preprocessing"])

    @logger.catch
    def postprocess(self, model_output, metadata):
        return self.postprocess_function(model_output, metadata, self.config["postprocessing"])

    @logger.catch
    def __call__(self, inputs: Union[Frame, list[Frame]]):
        # input must be a frame or a list of frames
        if isinstance(inputs, list):
            for frame in inputs:
                assert isinstance(frame, Frame)
        else:
            assert isinstance(inputs, Frame)

        model_inputs, metadata = self.preprocess_function(inputs, self.config["preprocessing"])
        model_outputs = self.forward(model_inputs)
        outputs = self.postprocess_function(model_outputs, metadata, self.config["postprocessing"])

        return outputs

    def forward(self, model_inputs):
        # compute ONNX Runtime output prediction
        if len(self.input_names) == 1:
            ort_inputs = {self.input_names[0]: model_inputs}
        else:
            ort_inputs = {name: value for name, value in zip(self.input_names, model_inputs)}

        model_outputs = self.session.run(None, ort_inputs)
        return model_outputs


class Frame:
    """
    abstraction class for an image (or video frame)
    """

    def __init__(self, input):
        self._path = None
        if isinstance(input, str):
            # input image path
            if not os.path.exists(input):
                logger.warning(f"cannot find input image path: {input}")
                raise RuntimeError(f"cannot find input image path: {input}")
            else:
                # TODO: handle when input is grayscale image
                self._path = input
                self._data = cv2.cvtColor(cv2.imread(input), cv2.COLOR_BGR2RGB)
        elif isinstance(input, np.ndarray):
            if input.dtype == np.uint8:
                # TODO: handle when input is grayscale, check channel
                self._data = input
            else:
                raise RuntimeError(
                    "input to Frame must be an image path or np.ndarray of type uint8"
                )

    def horizontal_flip(self, inplace=False):
        if inplace:
            self._data = self._data[:, ::-1, :]
        else:
            new_frame = self.copy()
            new_frame.horizontal_flip(inplace=True)
            return new_frame

    def jitter_color(self, inplace=False, hgain=5, sgain=30, vgain=30):
        hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
        hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
        hsv_augs = hsv_augs.astype(np.int16)
        img_hsv = cv2.cvtColor(self.data(), cv2.COLOR_RGB2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

        jittered_img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        if inplace:
            self._data = jittered_img
        else:
            return Frame(jittered_img)

    def path(self):
        return self._path

    def data(self):
        return self._data

    def rgb(self):
        return self._data

    def bgr(self):
        return cv2.cvtColor(self._data, cv2.COLOR_RGB2BGR)

    def height(self):
        return self._data.shape[0]

    def width(self):
        return self._data.shape[1]

    def shape(self):
        return self._data.shape

    def crop(self, bounding_box: BoundingBox, allow_clipping: bool = False):
        top_left = bounding_box.top_left().int()
        bottom_right = bounding_box.bottom_right().int()
        x0, y0 = top_left.x(), top_left.y()
        x1, y1 = bottom_right.x(), bottom_right.y()

        if not allow_clipping:
            if x0 >= 0 and y0 >= 0 and x1 <= self.width() and y1 <= self.height():
                return Frame(self.data()[y0:y1, x0:x1, :])
            else:
                logger.debug("fails to crop frame")
                logger.debug(f"frame info: {self}")
                logger.debug(f"bounding box info: {bounding_box}")
                logger.info(
                    "if the bounding box exceeds the image size, set allow_clipping to True to crop"
                )
                return None
        else:
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(x1, self.width())
            y1 = min(y1, self.height())
            return Frame(self.data()[y0:y1, x0:x1, :])

    def draw_point(self, point: Point, thickness: int = -1, marker="circle"):
        assert marker in MARKERS
        if marker == "circle":
            self._data = cv2.circle(
                self.data(),
                center=point.int().tuple(),
                radius=point.radius(),
                color=point.color().rgb(),
                thickness=thickness,
            )
        elif marker == "cross":
            self._data = cv2.drawMarker(
                img=self.data(),
                position=point.int().tuple(),
                color=point.color().rgb(),
                markerType=cv2.MARKER_CROSS,
                markerSize=point.radius(),
                thickness=thickness,
            )
        elif marker == "tilted_cross":
            self._data = cv2.drawMarker(
                img=self.data(),
                position=point.int().tuple(),
                color=point.color().rgb(),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=point.radius(),
                thickness=thickness,
            )
        elif marker == "star":
            self._data = cv2.drawMarker(
                img=self.data(),
                position=point.int().tuple(),
                color=point.color().rgb(),
                markerType=cv2.MARKER_STAR,
                markerSize=point.radius(),
                thickness=thickness,
            )
        elif marker == "diamond":
            self._data = cv2.drawMarker(
                img=self.data(),
                position=point.int().tuple(),
                color=point.color().rgb(),
                markerType=cv2.MARKER_DIAMOND,
                markerSize=point.radius(),
                thickness=thickness,
            )
        elif marker == "square":
            self._data = cv2.drawMarker(
                img=self.data(),
                position=point.int().tuple(),
                color=point.color().rgb(),
                markerType=cv2.MARKER_SQUARE,
                markerSize=point.radius(),
                thickness=thickness,
            )
        elif marker == "triangle_up":
            self._data = cv2.drawMarker(
                img=self.data(),
                position=point.int().tuple(),
                color=point.color().rgb(),
                markerType=cv2.MARKER_TRIANGLE_UP,
                markerSize=point.radius(),
                thickness=thickness,
            )
        elif marker == "triangle_down":
            self._data = cv2.drawMarker(
                img=self.data(),
                position=point.int().tuple(),
                color=point.color().rgb(),
                markerType=cv2.MARKER_TRIANGLE_DOWN,
                markerSize=point.radius(),
                thickness=thickness,
            )

    def draw_points(self, points: list[Point], thickness: int = -1, marker="circle"):
        for point in points:
            self.draw_point(point, thickness, marker)

    def draw_line(
        self,
        start_point: Point,
        end_point: Point,
        color: Color,
        thickness: int,
        draw_point: bool = True,
    ):
        if draw_point:
            self.draw_point(start_point)
            self.draw_point(end_point)

        self._data = cv2.arrowedLine(
            img=self.data(),
            pt1=start_point.int().tuple(),
            pt2=end_point.int().tuple(),
            color=color.rgb(),
            thickness=thickness,
        )

    def draw_segment(
        self,
        start_point: Point,
        end_point: Point,
        color: Color,
        thickness: int,
        draw_point: bool = True,
    ):
        if draw_point:
            self.draw_point(start_point)
            self.draw_point(end_point)

        self._data = cv2.line(
            img=self.data(),
            pt1=start_point.int().tuple(),
            pt2=end_point.int().tuple(),
            color=color.rgb(),
            thickness=thickness,
        )

    def draw_text(
        self,
        text: str,
        start_point: Point,
        color: Color,
        thickness: int = 2,
        font_scale: float = 0.75,
    ):
        self._data = cv2.putText(
            self.data(),
            text,
            start_point.int().tuple(),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=color.bgr(),
            thickness=thickness,
        )

    def draw_bounding_box(self, box: BoundingBox, allow_clipping: bool = False):
        # convert to int coordinates
        top_left = box.top_left().int()
        bottom_right = box.bottom_right().int()
        x0, y0 = top_left.x(), top_left.y()
        x1, y1 = bottom_right.x(), bottom_right.y()

        # clipping
        if allow_clipping:
            x0 = min(max(0, x0), self.width() - 1)
            x1 = min(max(0, x1), self.width() - 1)
            y0 = min(max(0, y0), self.height() - 1)
            y1 = min(max(0, y1), self.height() - 1)

        if (
            0 <= x0 < self.width()
            and 0 <= x1 < self.width()
            and 0 <= y0 < self.height()
            and 0 <= y1 < self.height()
        ):
            # valid box
            self._data = draw_rect(
                image=self.data(),
                points=[x0, y0, x1, y1],
                rgb=box.color().rgb(),
                label_transparency=box.label_transparency(),
                thickness=box.thickness(),
                labels=box.label(),
                label_rgb=box.label_color().rgb(),
                label_bg_rgb=box.label_background_color().rgb(),
                label_font_size=box.label_font_size(),
            )
        else:
            logger.warning("got invalid box when putting into the frame")
            logger.warning(f"bounding_box={box}")
            logger.warning(f"frame={self}")
            raise RuntimeError("got invalid box when putting into the frame")

    def draw_bounding_boxes(self, boxes: list[BoundingBox], allow_clipping: bool = False):
        for box in boxes:
            self.draw_bounding_box(box, allow_clipping)

    def __str__(self):
        return "Frame(height={}, width={})".format(self.height(), self.width())

    def save(self, path: str):
        ext = path.split(".")[-1]
        assert ext in ["jpg", "png", "JPG", "JPEG", "jpeg"]
        cv2.imwrite(path, self._data[:, :, ::-1])

    def copy(self):
        return Frame(np.copy(self.data()))

    def append_top(self, frame: Frame, inplace=False):
        # append another frame on top of this frame
        # width must be the same
        if frame.width() != self.width():
            raise RuntimeError("width must be the same to append on top")

        data = np.ascontiguousarray(np.concatenate([frame._data, self._data], axis=0))
        if inplace:
            self._data = data
            return
        else:
            return Frame(data)

    def append_bottom(self, frame: Frame, inplace=False):
        # append another frame at the bottom of this frame
        # width must be the same
        if frame.width() != self.width():
            raise RuntimeError("width must be the same to append at the bottom")

        data = np.ascontiguousarray(np.concatenate([self._data, frame._data], axis=0))
        if inplace:
            self._data = data
            return
        else:
            return Frame(data)

    def append_right(self, frame: Frame, inplace=False):
        # append another frame to the right of this frame
        # height must be the same
        if frame.height() != self.height():
            raise RuntimeError("height must be the same to append to the right")

        data = np.ascontiguousarray(np.concatenate([self._data, frame._data], axis=1))
        if inplace:
            self._data = data
            return
        else:
            return Frame(data)

    def append_left(self, frame: Frame, inplace=False):
        # append another frame to the left of this frame
        # height must be the same
        if frame.height() != self.height():
            raise RuntimeError("height must be the same to append to the left")

        data = np.ascontiguousarray(np.concatenate([frame._data, self._data], axis=1))
        if inplace:
            self._data = data
            return
        else:
            return Frame(data)

    def resize(
        self,
        new_width: int,
        new_height: int,
        keep_ratio: bool = False,
        pad_constant: int = 0,
        pad_position: str = "fixed",
        interpolation="linear",
    ):
        assert interpolation in INTERPOLATIONS
        if keep_ratio:
            # need to keep aspect ratio
            if pad_constant is not None:
                assert isinstance(pad_constant, int)
                assert 0 <= pad_constant <= 255
            else:
                pad_constant = np.random.randint(low=0, high=256)

            assert pad_position in ["fixed", "random"]

            new_frame_data = pad_constant * np.ones((new_height, new_width, 3), dtype=np.uint8)
            ratio = min(new_height / self.height(), new_width / self.width())
            sub_width = int(ratio * self.width())
            sub_height = int(ratio * self.height())
            sub_data = cv2.resize(
                self.data(),
                (sub_width, sub_height),
                interpolation=INTERPOLATIONS[interpolation],
            )
            if pad_position == "fixed":
                # put the image to the top left
                new_frame_data[:sub_height, :sub_width, :] = sub_data
            else:
                # randomly put inside
                if new_width - sub_width > 0:
                    start_x = np.random.randint(low=0, high=new_width - sub_width)
                else:
                    start_x = 0
                if new_height - sub_height > 0:
                    start_y = np.random.randint(low=0, high=new_height - sub_height)
                else:
                    start_y = 0

                new_frame_data[
                    start_y : start_y + sub_height, start_x : start_x + sub_width, :
                ] = sub_data

            return Frame(new_frame_data), ratio
        else:
            # no need to keep aspect ratio
            new_frame_data = cv2.resize(
                self.data(),
                (new_width, new_height),
                interpolation=INTERPOLATIONS[interpolation],
            )
            return Frame(new_frame_data)

    def masking(self, mask: SegmentMask, inplace: bool = False):
        """
        Apply a mask to the frame
        """

        if inplace:
            self._data[:, :, :] = (self._data * mask.data()).astype(np.uint8)
        else:
            return Frame((self._data * mask.data()).astype(np.uint8))

    def overlay_mask(self, mask: SegmentMask, alpha: float, inplace: bool = False):
        """
        Overlay mask with color on top the image
        """
        assert alpha >= 0 and alpha <= 1

        R, G, B = mask.color().rgb()
        color_mask = np.concatenate([mask.data() * R, mask.data() * G, mask.data() * B], axis=2)

        # combine
        data = ((1 - alpha) * self._data + alpha * color_mask).astype(np.uint8)

        if inplace:
            self._data = data
        else:
            return Frame(data)

    def show(self):
        """
        Show this frame and exit when any key is pressed
        """
        cv2.imshow("Frame", self.bgr())
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class BoundingBox:
    """
    Abstraction for bounding box, including color and line style for visualization
    """

    def __init__(
        self,
        top_left: Point,
        bottom_right: Point,
        confidence: float,
        color=Color(),
        thickness=1,
        label=None,
        label_color=Color(0, 0, 0),
        label_background_color=Color(255, 255, 255),
        label_font_size=1,
        label_transparency=0,
    ):
        # convention: x corresponds to width, y corresponds to height
        assert isinstance(top_left, Point)
        assert isinstance(bottom_right, Point)

        assert bottom_right.x() > top_left.x()
        assert bottom_right.y() > top_left.y()
        assert 0 <= confidence <= 1

        self._top_left = top_left
        self._bottom_right = bottom_right
        self._color = color
        self._thickness = thickness
        self._label = label
        self._label_color = label_color
        self._label_background_color = label_background_color
        self._label_font_size = label_font_size
        self._label_transparency = label_transparency
        self._confidence = confidence

    @staticmethod
    def from_xywh(
        x: Union[int, float],
        y: Union[int, float],
        width: int,
        height: int,
        confidence: float = 1.0,
        color=Color(),
        thickness=1,
        label=None,
        label_color=Color(0, 0, 0),
        label_background_color=Color(255, 255, 255),
        label_font_size=1,
        label_transparency=0,
    ) -> BoundingBox:
        """return an instance of BoundingBox from (x, y, width, height)"""

        return BoundingBox(
            top_left=Point(x, y),
            bottom_right=Point(x + width, y + height),
            confidence=confidence,
            color=color,
            thickness=thickness,
            label=label,
            label_color=label_color,
            label_background_color=label_background_color,
            label_font_size=label_font_size,
            label_transparency=label_transparency,
        )

    @staticmethod
    def from_x0y0x1y1(
        x0: Union[int, float],
        y0: Union[int, float],
        x1: Union[int, float],
        y1: Union[int, float],
        confidence: float = 1.0,
        color=Color(),
        thickness=1,
        label=None,
        label_color=Color(0, 0, 0),
        label_background_color=Color(255, 255, 255),
        label_font_size=1,
        label_transparency=0,
    ) -> BoundingBox:
        """return an instance of BoundingBox from (x0, y0, x1, y1)"""

        return BoundingBox(
            top_left=Point(x0, y0),
            bottom_right=Point(x1, y1),
            confidence=confidence,
            color=color,
            thickness=thickness,
            label=label,
            label_color=label_color,
            label_background_color=label_background_color,
            label_font_size=label_font_size,
            label_transparency=label_transparency,
        )

    def translate(self, reference_point: Point, inplace=False):
        top_left = self.top_left() + reference_point
        bottom_right = self.bottom_right() + reference_point
        if inplace:
            self._top_left = top_left
            self._bottom_right = bottom_right
        else:
            return BoundingBox(
                top_left=top_left,
                bottom_right=bottom_right,
                confidence=self.confidence(),
                color=self.color(),
                thickness=self.thickness(),
                label=self.label(),
                label_color=self.label_color(),
                label_background_color=self.label_background_color(),
                label_font_size=self.label_font_size(),
                label_transparency=self.label_transparency(),
            )

    def clip(self, height: int, width: int, inplace=False):
        top_left_x = min(max(0, self._top_left.x()), width)
        top_left_y = min(max(0, self._top_left.y()), height)
        bottom_right_x = min(max(0, self._bottom_right.x()), width)
        bottom_right_y = min(max(0, self._bottom_right.y()), height)

        if bottom_right_x <= top_left_x or bottom_right_y <= top_left_y:
            logger.warning("failed to perform BoundingBox.clip() because top_left > bottom_right")
            logger.warning(f"bounding box being clipped: {self}")
            logger.warning(f"image height={height}, image width={width}")
            raise RuntimeError(
                "failed to perform BoundingBox.clip() because top_left > bottom_right"
            )

        if inplace:
            self._top_left.set_x(top_left_x)
            self._top_left.set_y(top_left_y)
            self._bottom_right.set_x(bottom_right_x)
            self._bottom_right.set_y(bottom_right_y)
        else:
            new_box = self.copy()
            new_box.top_left().set_x(top_left_x)
            new_box.top_left().set_y(top_left_y)
            new_box.bottom_right().set_x(bottom_right_x)
            new_box.bottom_right().set_y(bottom_right_y)
            return new_box

    def xywh(
        self,
    ) -> tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]]:
        """return the coordinates in the format (x, y, width, height)"""
        return (
            self.top_left().x(),
            self.top_left().y(),
            self.width(),
            self.height(),
        )

    def x0y0x1y1(
        self,
    ) -> tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]]:
        """return the coordinates in the format (x0, y0, x1, y1)"""
        return (
            self.top_left().x(),
            self.top_left().y(),
            self.bottom_right().x(),
            self.bottom_right().y(),
        )

    def iou(self, box: BoundingBox) -> float:
        """
        compute the Intersection over Union (IoU) with a given box
        """
        # Determine the coordinates of the intersection rectangle
        x_left = max(self.top_left().x(), box.top_left().x())
        y_top = max(self.top_left().y(), box.top_left().y())
        x_right = min(self.bottom_right().x(), box.bottom_right().x())
        y_bottom = min(self.bottom_right().y(), box.bottom_right().y())

        # intersection
        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
        # union
        union_area = self.area() + box.area() - intersection_area

        # iou
        return intersection_area / union_area

    def intersection(self, box: BoundingBox) -> BoundingBox:
        """return the intersection with another box"""
        # Determine the coordinates of the intersection rectangle
        x_left = max(self.top_left().x(), box.top_left().x())
        y_top = max(self.top_left().y(), box.top_left().y())
        x_right = min(self.bottom_right().x(), box.bottom_right().x())
        y_bottom = min(self.bottom_right().y(), box.bottom_right().y())
        top_left = Point(x_left, y_top)
        bottom_right = Point(x_right, y_bottom)

        return BoundingBox(top_left=top_left, bottom_right=bottom_right, confidence=1.0)

    def union(self, box: BoundingBox) -> BoundingBox:
        """compute the union box of 2 boxes"""
        x_left = min(self.top_left().x(), box.top_left().x())
        y_top = min(self.top_left().y(), box.top_left().y())
        x_right = max(self.bottom_right().x(), box.bottom_right().x())
        y_bottom = max(self.bottom_right().y(), box.bottom_right().y())
        top_left = Point(x_left, y_top)
        bottom_right = Point(x_right, y_bottom)
        confidence = max(self.confidence(), box.confidence())

        return BoundingBox(top_left=top_left, bottom_right=bottom_right, confidence=confidence)

    def area(self):
        return self.width() * self.height()

    def scale(self, scale: float):
        new_width = self.width() * scale
        new_height = self.height() * scale
        if new_width < 1 or new_height < 1:
            logger.warning(f"ignore bounding box scaling due invalid output bounding box")
            return

        top_left = self.center() - Point(new_width, new_height) / Point(2.0, 2.0)
        bottom_right = self.center() + Point(new_width, new_height) / Point(2.0, 2.0)

        return BoundingBox(
            top_left=top_left.int(),
            bottom_right=bottom_right.int(),
            confidence=self.confidence(),
            color=self.color(),
            thickness=self.thickness(),
            label=self.label(),
            label_color=self.label_color(),
            label_background_color=self.label_background_color(),
            label_font_size=self.label_font_size(),
            label_transparency=self.label_transparency(),
        )

    def top_left(self):
        return self._top_left

    def bottom_right(self):
        return self._bottom_right

    def center(self):
        return self._top_left + Point(self.width(), self.height()) / Point(2.0, 2.0)

    def confidence(self):
        return self._confidence

    def height(self):
        return self.bottom_right().y() - self.top_left().y()

    def width(self):
        return self.bottom_right().x() - self.top_left().x()

    def thickness(self):
        return self._thickness

    def set_thickness(self, thickness):
        self._thickness = thickness

    def label(self):
        return self._label

    def set_label(self, label: str):
        self._label = label

    def color(self):
        return self._color

    def set_color(self, color: Color):
        self._color = color

    def label_color(self):
        return self._label_color

    def set_label_color(self, color: Color):
        self._label_color = color

    def label_background_color(self):
        return self._label_background_color

    def set_label_background_color(self, color: Color):
        self._label_background_color = color

    def label_font_size(self):
        return self._label_font_size

    def set_label_font_size(self, font_size: int):
        self._label_front_size = font_size

    def label_transparency(self):
        return self._label_transparency

    def set_label_transparency(self, transparency):
        self._label_transparency = transparency

    def __str__(self):
        return (
            "BoundingBox(height={}, width={}, confidence={}, ".format(
                self.height(), self.width(), self.confidence()
            )
            + "top_left={}, bottom_right={}), ".format(self.top_left(), self.bottom_right())
            + "label={}, label_color={}), ".format(self.label(), self.label_color())
            + "label_background_color={}, ".format(self.label_background_color())
            + "label_font_size={}, ".format(self.label_font_size())
            + "label_transparency={})".format(self.label_transparency())
        )

    def copy(self) -> BoundingBox:
        return BoundingBox(
            top_left=self.top_left(),
            bottom_right=self.bottom_right(),
            confidence=self.confidence(),
            color=self.color(),
            thickness=self.thickness(),
            label=self.label(),
            label_color=self.label_color(),
            label_background_color=self.label_background_color(),
            label_font_size=self.label_font_size(),
            label_transparency=self.label_transparency(),
        )


class Point:
    """
    abstraction for a point
    """

    def __init__(self, x: Any, y: Any, color=Color(), radius=2):
        # note x, y correspond to points on the width and height axis
        assert radius > 0
        self._x = x
        self._y = y
        self._color = color
        self._radius = radius

    def copy(self):
        return Point(self.x(), self.y(), self.color(), self.radius())

    def tuple(self) -> Tuple[Any, Any]:
        return (self._x, self._y)

    def int(self, inplace=False) -> Point:
        if inplace:
            self._x = int(self._x)
            self._y = int(self._y)
        else:
            new_point = self.copy()
            new_point.set_x(int(self.x()))
            new_point.set_y(int(self.y()))
            return new_point

    def color(self):
        return self._color

    def set_color(self, color: Color):
        self._color = color

    def radius(self):
        return self._radius

    def set_radius(self, r: int):
        assert r > 0
        self._radius = r

    def x(self):
        return self._x

    def set_x(self, x):
        self._x = x

    def y(self):
        return self._y

    def set_y(self, y):
        self._y = y

    def translate(self, reference: Point):
        """
        change the coordinate system so that the current origin (0, 0) has a coordinate equal to
        `reference` point in the new system

        this is useful when mapping from a coorindate in a cropped image to the coorindate in
        the original image --> we need to pass the top left point as the reference point
        """
        return Point(self.x() + reference.x(), self.y() + reference.y())

    def __str__(self):
        return "Point(x={}, y={}, color={}, radius={})".format(
            self.x(), self.y(), self.color(), self.radius()
        )

    def __add__(self, other: Point):
        """
        Add two points.
        """
        new_point = self.copy()
        new_point.set_x(self.x() + other.x())
        new_point.set_y(self.y() + other.y())
        return new_point

    def __sub__(self, other: Point):
        """
        Subtract two points.
        """
        new_point = self.copy()
        new_point.set_x(self.x() - other.x())
        new_point.set_y(self.y() - other.y())
        return new_point

    def __mul__(self, other: Point):
        """
        Multiply two points.
        """
        new_point = self.copy()
        new_point.set_x(self.x() * other.x())
        new_point.set_y(self.y() * other.y())
        return new_point

    def __truediv__(self, other: Point):
        """
        Divide two points.
        """
        new_point = self.copy()
        new_point.set_x(self.x() / other.x())
        new_point.set_y(self.y() / other.y())
        return new_point

    def __pow__(self, scalar: int):
        """
        Compute exponential each coordinate.
        """
        new_point = self.copy()
        new_point.set_x(self.x() ** scalar)
        new_point.set_y(self.y() ** scalar)
        return new_point

    def __abs__(self):
        """
        Compute absolute values of point coordinates.
        """
        new_point = self.copy()
        new_point.set_x(abs(self.x()))
        new_point.set_y(abs(self.y()))
        return new_point

    def distance(self, other: Point):
        """
        Compute euclidean distance.
        """
        delta = (self - other) ** 2
        return np.sqrt(delta.x() + delta.y())

    def magnitude(self) -> float:
        return np.sqrt(self.x() ** 2 + self.y() ** 2)


class KeyPoint(Point):
    """
    Abstraction for keypoint. A keypoint is a point with confidence score
    """

    def __init__(self, x: int, y: int, confidence: float, color: Color(), radius: int = 2):
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
        self._confidence = confidence
        super().__init__(x, y, color, radius)

    def copy(self):
        return KeyPoint(self.x(), self.y(), self.confidence(), self.color(), self.radius())

    def set_confidence(self, confidence: float):
        assert 0 <= confidence <= 1
        self._confidence = confidence

    def confidence(self):
        return self._confidence

    def translate(self, reference: Point):
        """
        change the coordinate system so that the current origin (0, 0) has a coordinate equal to
        `reference` point in the new system

        this is useful when mapping from a coorindate in a cropped image to the coorindate in
        the original image --> we need to pass the top left point as the reference point
        """
        return KeyPoint(
            self.x() + reference.x(),
            self.y() + reference.y(),
            self._confidence,
            self.color(),
            self.radius(),
        )

    def __str__(self):
        return "KeyPoint(x={}, y={}, confidence={:.2f} %, color={}, radius={})".format(
            self.x(),
            self.y(),
            100 * self.confidence(),
            self.color(),
            self.radius(),
        )


class SegmentMask:
    """
    Datastructure for segmentation mask
    """

    def __init__(self, data: Any, score: float, color: Color = Color()):
        # must be numpy array
        assert isinstance(data, np.ndarray)
        # must be 2 dimension
        assert data.ndim == 2
        # must be between [0,1]

        assert data.min() >= 0 and data.max() <= 1

        # we save by extending last dim to H x W x 1
        self._data = np.expand_dims(data, axis=-1)

        # original score or confidence
        self._score = score
        self._color = color

    def data(self):
        return self._data

    def color(self):
        return self._color

    def set_color(self, color: Color):
        self._color = color

    def score(self):
        return self._score

    def as_frame(self) -> Frame:
        """
        Return mask as a Frame object
        """
        # replicate mask to 3 channels
        data = np.concatenate([self.data()] * 3, axis=2)

        # convert to uint8
        data = (data * 255).astype(np.uint8)

        return Frame(data)


class BinaryBlob:
    """
    abstraction for binary blob storage
    """

    def __init__(self, binary_file: str, index_file: str, mode="r"):
        assert mode in ["r", "w"]
        self._mode = "write" if mode == "w" else "read"

        if mode == "w":
            # writing mode
            self._fid = open(binary_file, "wb")
            self._idx_fid = open(index_file, "w")
            self._indices = set()
        else:
            assert os.path.exists(binary_file)
            assert os.path.exists(index_file)

            # read index file
            with open(index_file, "r") as fid:
                content = fid.read().split("\n")[:-1]

            self._index_content = {}
            self._indices = set()
            for row in content:
                sample_idx, byte_pos, byte_length, need_conversion = row.split(",")
                self._index_content[int(sample_idx)] = (
                    int(byte_pos),
                    int(byte_length),
                    bool(int(need_conversion)),
                )
                self._indices.add(int(sample_idx))

            # open binary file
            self._fid = open(binary_file, "rb")
            self._fid.seek(0, 0)
            self._idx_fid = None

            # sorted indices
            self._sorted_indices = list(self._indices)
            self._sorted_indices.sort()

        self._cur_index = -1

    def __iter__(self):
        self._cur_index = -1
        return self

    def __next__(self):
        self._cur_index += 1
        if self._cur_index < len(self):
            return self.__getitem__(self._cur_index)
        raise StopIteration

    def __getitem__(self, i: int):
        if self._mode == "write":
            raise RuntimeError(
                "__getitem__ is not supported when BinaryBlob is opened in write mode"
            )

        if i >= len(self):
            raise RuntimeError(f"index {i} is out of range: [0 - {len(self)})")
        idx = self._sorted_indices[i]
        return self.read_index(idx)

    def __len__(self):
        if self._mode == "write":
            raise RuntimeError("__len__ is not supported when BinaryBlob is opened in write mode")
        return len(self._sorted_indices)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write_index(self, index: int, content):
        assert isinstance(index, int)
        if self._mode == "write":
            # allow writing
            try:
                # check if index existence
                if index in self._indices:
                    raise RuntimeError(f"Given index={index} has been occuppied. Cannot write")

                # convert to byte string
                if not isinstance(content, bytes):
                    content = dill.dumps(content)
                    # flag to mark whether serialization/deserialization is
                    # needed
                    converted = 1
                else:
                    converted = 0

                # log position before writing
                current_pos = self._fid.tell()
                # write byte string
                self._fid.write(content)
                # write metadata information
                self._idx_fid.write(f"{index},{current_pos},{len(content)},{converted}\n")

                # keep track of index
                self._indices.add(index)

            except Exception as error:
                self.close()
                raise error
        else:
            # raise error
            self.close()
            raise RuntimeError("BinaryBlob was opened in reading mode. No writing allowed")

    def read_index(self, index: int):
        assert isinstance(index, int)
        assert index >= 0

        if self._mode == "read":
            if index not in self._indices:
                self.close()
                raise RuntimeError(f"Given index={index} does not exist in BinaryBlob")

            # pos is the starting position we need to seek
            target_pos, length, need_conversion = self._index_content[index]
            # we need to compute seek parameter
            delta = target_pos - self._fid.tell()
            # seek to target_pos
            self._fid.seek(delta, 1)

            # read `length` number of bytes
            item = self._fid.read(length)
            # deserialize if needed
            if need_conversion:
                item = dill.loads(item)
            return item
        else:
            self.close()
            raise RuntimeError("BinaryBlob was opened in writing mode. No reading allowed")

    def close(self):
        if self._fid is not None:
            self._fid.close()
        if self._idx_fid is not None:
            self._idx_fid.close()
