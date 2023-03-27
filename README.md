# CVInfer - A Computer Vision Inference Toolkit 

This provides an interface for collaboration between model developer and model deployer.

The principle is that, the model deployer doesn't need to know the logic behind a model.

He only follows the contract (interface) defined via this library. 

When a model developer wants to deploy a model, he needs to generate the following artifacts:

- `onnx file`: the core deep learning model converted to ONNX format
- `processor.py`: this contains 2 functions: `preprocess()` and `postprocess()`, which hold the logics for preprocessing and postprocessing 
- `requirements.txt`:  this contains all the dependencies required to use the artifacts (a.k.a dependencies in preprocessing and postprocessing steps)
- `configuration.json`: this contains referenced values for the configuration values such as confidence threshold or input image size expected by a model.
This json file should contain 2 sections: one called `preprocessing` and one called `postprocessing`, which contain the config values for the pre and post processing steps. 
- `README.md`: a documentation that describes dependencies' installation and configuration explanations. 


This library also provides common interfaces and io utilities for CV deployment.

## Installation

Install dependencies in the `requirements.txt` then run `pip install -e . ` to install this library in development mode.


## `cvinfer.common` module

This module provides common interfaces for CV tasks:

- `cvinfer.common.Frame`: abstraction for an image or a video frame.
  This class contains popular image manipulation methods such as resizing, horizontal flipping, color jittering, bounding box drawing, saving etc.
- `cvinfer.common.BoundingBox`: abstraction for a bounding box
  A bounding box contains not only coordinate information but also the color, confidence score, label etc. 
- `cvinfer.common.Color`: abstraction for color
- `cvinfer.common.Point`: abstraction for a point
- `cvinfer.common.KeyPoint`: a keypoint is a Point with confidence score
- `cvinfer.common.OnnxModel`: abstraction to use an ONNX model. See more in the `examples`. 

Please take a look at source code to see what kind of methods are supported by each class

## `cvinfer.io` module

- `VideoReader`: convenient class for video reader
- `RTSPReader`: abstraction to read from RTSP stream
- `WebcamReader`: abstraction for webcam
- `VideoWriter`: a video writer convenient class.

Please take a look at `examples/io_examples.py` for the demo usage of IO classes. 


## Convention for artifact generation 
This provides description of the interface that should be used by the model developer when preparing the artifacts for deployment

Example of artifacts generation can be found in under `examples` directory.

**Rule 1**: Every preprocessing function should have the following signature:

```python
def preprocess(frame: cvinfer.common.Frame, config: dict):
```

The 1st argument to `preprocess` should be a `cvinfer.common.Frame` object or a list of `cvinfer.common.Frame` objects (for models that utilized multiple Frames). 

`Frame` is a class that abstracts an image, which is defined under `cvinfer.common.Frame`.  

Basically, `cvinfer.common.Frame` defines an interface on what kind of values to be expected from an image and what kind of operations that we could perform. 

The 2nd argument to `preprocess` should be a dictionary that contains the necessary configuration values needed to preprocess the image. 

These configuration values are those specified under the `preprocessing` section of the `configuration.json` file mentioned above in the artifacts. 

For example, an object detection model often works with specific image size. We could specify the image resize for resizing in the `preprocessing` section of the json file. 

The `preprocessing` function should return 2 values: the 1st value should be the preprocessed inputs that will be passed to the ONNX model and the 2nd value could be any information that will be used later by the postprocessing function. For example, we might want to pass the original image size in this 2nd returned variable in object detection. 

Note that the 1st variable returned by the preprocessing function will be passed to the ONNX model. If your model takes only 1 input, it should be a numpy array of appropriate dtype and shape. If your model expects 2 inputs, it should be a list of 2 numpy arrays. 

**Rule 2**: Every postprocessing function should have the following signature:

```python
def postprocess(model_output, metadata, config)
```

The 1st argument to `postprocess` is the output from the ONNX model. Note that if your deep learning model only returns 1 array, `model_output` will be a list of 1 numpy array. 

If your model returns 2 arrays, `model_output` will be a list of 2 numpy arrays. That is, `model_output` is exactly what returned by running the ONNX model. 

The 2nd argument to `postprocess` is the 2nd variable returned by the `preprocess` function. 

This could be any value that we need from the preprocessing step such as the original image size and so on. 

The 3rd argument to `postprocess` is the configuration dictionary that contains any configuration needed for postprocessing. For example, in object detection, we need confidence threshold and non-maximum suppression threshold to perform postprocessing. 

We should access these thresholds via the 3rd argument. The keys should be the same keys that appear under the `postprocessing` section in the `configuration.json` file.


## Authors
Dat Tran (viebboy@gmail.com)
