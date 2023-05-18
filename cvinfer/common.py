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


logger.warning('Convention for BoundingBox: (x, y) corresponds to coordinates on the (width, height) dimension')

INTERPOLATIONS = {
    'cubic': cv2.INTER_CUBIC,
    'linear': cv2.INTER_LINEAR,
    'nearest': cv2.INTER_NEAREST,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4,
}


def load_module(module_file, attribute, module_name=None):
    if module_name is None:
        module_name = ''.join(random.sample(string.ascii_letters, 10))

    try:
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error1:
        try:
            # try to append directory that contains module_file to path
            logger.debug(f'fails to import attribute {attribute} from module {module_file}')
            module_path = os.path.dirname(os.path.abspath(module_file))
            logger.debug(f'trying to append {module_path} to sys.path to fix this issue')
            sys.path.append(module_path)

            spec = importlib.util.spec_from_file_location(module_name, module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as error2:
            logger.error(f'ERROR when trying to load from {module_file} without appending parent dir')
            logger.error(str(error1))
            logger.error(f'ERROR when trying to load from {module_file} WITH parent dir appended to sys.path')
            logger.error(str(error2))
            raise error1

    if hasattr(module, attribute):
        return getattr(module, attribute)
    else:
        raise RuntimeError(f'Cannot find attribute {attribute} in the given module at {module_file}')

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
            duration = '{:.6f}'.format(stop - self.start)
        else:
            duration = '{:.2f}'.format(stop - self.start)
        self.logger.info(f'{self.function_name} took {duration} seconds')


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
        return '(R={}, G={}, B={})'.format(*self.rgb())


class OnnxModel:
    """
    base implementation of the onnx inference model
    """
    def __init__(
        self,
        onnx_path,
        processor_path=None,
        configuration_path=None,
        execution_provider='CPUExecutionProvider',
    ):
		# load preprocessing function
        if processor_path is None:
            processor_path = onnx_path.replace('.onnx', '.processor.py')

        if not os.path.exists(processor_path):
            raise FileNotFoundError(processor_path)

        self.preprocess_function = load_module(processor_path, 'preprocess')
        self.postprocess_function = load_module(processor_path, 'postprocess')

        #load onnx model from onnx_path
        avail_providers = ORT.get_available_providers()
        logger.info('all available ExecutionProviders are:')
        for idx, provider in enumerate(avail_providers):
            logger.info(f'\t {provider}')

        logger.info(f'trying to run with execution provider: {execution_provider}')
        self.session =  ORT.InferenceSession(onnx_path, providers=[execution_provider,])

        self.input_names = [input_.name for input_ in self.session.get_inputs()]

        if configuration_path is None:
            configuration_path = onnx_path.replace('.onnx', '.configuration.json')

        if not os.path.exists(configuration_path):
            raise FileNotFoundError(configuration_path)

        with open(configuration_path, 'r') as fid:
            # self.config is a dictionary
            self.config = json.loads(fid.read())

    @logger.catch
    def preprocess(self, inputs: Union[Frame, list[Frame]]):
        assert isinstance(inputs, Frame)
        return self.preprocess_function(inputs, self.config['preprocessing'])

    @logger.catch
    def postprocess(self, model_output, metadata):
        return self.postprocess_function(model_output, metadata, self.config['postprocessing'])

    @logger.catch
    def __call__(self, inputs: Union[Frame, list[Frame]]):
        # input must be a frame or a list of frames
        if isinstance(inputs, list):
            for frame in inputs:
                assert isinstance(frame, Frame)
        else:
            assert isinstance(inputs, Frame)

        # calling preprocess
        model_inputs, metadata = self.preprocess_function(
            inputs,
            self.config['preprocessing']
        )

        # compute ONNX Runtime output prediction
        if len(self.input_names) == 1:
            ort_inputs = {self.input_names[0]: model_inputs}
        else:
            ort_inputs = {name: value for name, value in zip(self.input_names, model_inputs)}

        model_outputs = self.session.run(None, ort_inputs)

        outputs = self.postprocess_function(
            model_outputs,
            metadata,
            self.config['postprocessing']
        )

        return outputs


class Frame:
    """
    abstraction class for an image (or video frame)
    """
    def __init__(self, input):
        self._path = None
        if isinstance(input, str):
            # input image path
            if not os.path.exists(input):
                logger.warning(f'cannot find input image path: {input}')
                raise RuntimeError(f'cannot find input image path: {input}')
            else:
                #TODO: handle when input is grayscale image
                self._path = input
                self._data = cv2.imread(input)[:, :, ::-1]
        elif isinstance(input, np.ndarray):
            if input.dtype == np.uint8:
                #TODO: handle when input is grayscale, check channel
                self._data = input
            else:
                raise RuntimeError('input to Frame must be an image path or np.ndarray of type uint8')

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
        return self._data[:, :, ::-1]

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
            if x0 >= 0 and y0 >= 0 and x1 < self.width() and y1 < self.height():
                return Frame(self.data()[y0: y1, x0: x1, :])
            else:
                logger.debug(f'fails to crop frame')
                logger.debug(f'frame info: {self}')
                logger.debug(f'bounding box info: {bounding_box}')
                logger.info(f'if the bounding box exceeds the image size, set allow_clipping to True to crop')
                return None
        else:
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(x1, self.width())
            y1 = min(y1, self.height())
            return Frame(self.data()[y0: y1, x0: x1, :])

    def draw_point(self, point: Point):
        self._data = cv2.circle(
            self.data().copy(),
            center=point.int().tuple(),
            radius=point.radius(),
            color=point.color().rgb(),
            thickness=-1,
        )

    def draw_points(self, points: list[Point]):
        for point in points:
            self.draw_point(point)

    def draw_line(
        self,
        start_point: Point,
        end_point: Point,
        color: Color,
        thickness: int,
        draw_point: bool=True,
    ):
        if draw_point:
            self.draw_point(start_point)
            self.draw_point(end_point)

        self._data = cv2.arrowedLine(
            img=self.data().copy(),
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
        draw_point: bool=True,
    ):
        if draw_point:
            self.draw_point(start_point)
            self.draw_point(end_point)

        self._data = cv2.line(
            img=self.data().copy(),
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
            font=cv2.FONT_HERSHEY_SIMPLEX,
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
            x0 = min(max(0, x0), self.width() -1)
            x1 = min(max(0, x1), self.width() -1)
            y0 = min(max(0, y0), self.height() -1)
            y1 = min(max(0, y1), self.height() -1)

        if 0 <= x0 < self.width() and 0 <= x1 < self.width() and 0 <= y0 < self.height() and 0 <= y1 < self.height():
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
            logger.warning('got invalid box when putting into the frame')
            logger.warning(f'bounding_box={box}')
            logger.warning(f'frame={self}')
            raise RuntimeError('got invalid box when putting into the frame')

    def draw_bounding_boxes(self, boxes: list[BoundingBox], allow_clipping: bool = False):
        for box in boxes:
            self.draw_bounding_box(box, allow_clipping)

    def __str__(self):
        return 'Frame(height={}, width={})'.format(self.height(), self.width())

    def save(self, path: str):
        ext = path.split('.')[-1]
        assert ext in ['jpg', 'png', 'JPG', 'JPEG', 'jpeg']
        cv2.imwrite(path, self._data[:, :, ::-1])

    def copy(self):
        return Frame(np.copy(self.data()))

    def resize(
        self,
        new_width: int,
        new_height: int,
        keep_ratio: bool = False,
        pad_constant: int = 0,
        pad_position: str = 'fixed',
        interpolation='linear'
    ):
        assert interpolation in INTERPOLATIONS
        if keep_ratio:
            # need to keep aspect ratio
            if pad_constant is not None:
                assert isinstance(pad_constant, int)
                assert 0 <= pad_constant <= 255
            else:
                pad_constant = np.random.randint(low=0, high=256)

            assert pad_position in ['fixed', 'random']

            new_frame_data = pad_constant * np.ones((new_height, new_width, 3), dtype=np.uint8)
            ratio = min(new_height / self.height(), new_width / self.width())
            sub_width = int(ratio * self.width())
            sub_height = int(ratio * self.height())
            sub_data = cv2.resize(
                self.data(),
                (sub_width, sub_height),
                interpolation=INTERPOLATIONS[interpolation],
            )
            if pad_position == 'fixed':
                # put the image to the top left
                new_frame_data[:sub_height, :sub_width, :] = sub_data
            else:
                # randomly put inside
                if new_width - sub_width > 0:
                    start_x = np.random.randint(low=0, high = new_width - sub_width)
                else:
                    start_x = 0
                if new_height - sub_height > 0:
                    start_y = np.random.randint(low=0, high = new_height - sub_height)
                else:
                    start_y = 0

                new_frame_data[start_y: start_y + sub_height, start_x: start_x + sub_width, :] = sub_data

            return Frame(new_frame_data), ratio
        else:
            # no need to keep aspect ratio
            new_frame_data = cv2.resize(
                self.data(),
                (new_width, new_height),
                interpolation=INTERPOLATIONS[interpolation]
            )
            return Frame(new_frame_data)


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

    def scale(self, scale: float):
        new_width = self.width() * scale
        new_height = self.height() * scale
        if new_width < 1 or new_height < 1:
            logger.warning(f'ignore bounding box scaling due invalid output bounding box')
            return

        top_left = self.center() - Point(new_width, new_height) / Point(2.0, 2.0)
        bottom_right = self.center() + Point(new_width, new_height) / Point(2.0, 2.0)

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

    def top_left(self):
        return self._top_left

    def bottom_right(self):
        return self._bottom_right

    def center(self):
        return self._top_left + Point(
            self.width(), self.height()) / Point(2.0, 2.0)

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
            'BoundingBox(height={}, width={}, confidence={}, '.format(self.height(), self.width(), self.confidence()) +
            'top_left={}, bottom_right={}), '.format(self.top_left(), self.bottom_right()) +
            'label={}, label_color={}), '.format(self.label(), self.label_color()) +
            'label_background_color={}, '.format(self.label_background_color()) +
            'label_font_size={}, '.format(self.label_font_size()) +
            'label_transparency={})'.format(self.label_transparency())
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

    def int(self) -> Point:
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
        return 'Point(x={}, y={}, color={}, radius={})'.format(
            self.x(),
            self.y(),
            self.color(),
            self.radius()
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
        return np.sqrt(self.x()**2 + self.y()**2)


class KeyPoint(Point):
    """
    Abstraction for keypoint. A keypoint is a point with confidence score
    """
    def __init__(self, x: int, y: int, confidence: float, color: Color(), radius: int=2):
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
        self._confidence = confidence
        super().__init__(x, y, color, radius)

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
        return 'KeyPoint(x={}, y={}, confidence={:.2f} %, color={}, radius={})'.format(
            self.x(),
            self.y(),
            100 * self.confidence(),
            self.color(),
            self.radius(),
        )
