# CVInfer - A Computer Vision Inference Toolkit 

This provides an interface for collaboration between model developer and model deployer.

The principle is that, the model deployer doesn't need to know the logic behind a model.

He only follows the contract (interface) defined via this library. 

When a model developer wants to deploy a model, he needs to generate the following artifacts:

- `model.onnx`: the core deep learning model converted to ONNX format
- `model.preprocess`: this is a serialized function (by dill) that encapsulates all the preprocessing steps.
- `model.postprocess`: this is a serialized function (by dill) that encapsulates all the postprocessing steps.
- `model.requirements.txt`:  this contains all the dependencies required to use the artifacts (a.k.a dependencies in preprocessing and postprocessing steps)
- `model.configuration.json`: this contains referenced values for the configuration values such as confidence threshold or input image size expected by a model.
This json file should contain 2 sections: one called `preprocessing` and one called `postprocessing`, which contain the config values for the pre and post processing steps. 
- `readme.md`: a documentation that explains the meaning of each configuration value


This library also provides common interfaces and io utilities for CV deployment.


## Installation

Install dependencies in the `requirements.txt` first then run `pip install -e . ` to install this library from source code.

## Interface For Model Developer - Artifacts generation
This provides description of the interface that should be used by the model developer when implementing artifact generation asset (converter script)

Example of artifacts generation can be found in under `examples` directory.


**Rule 1**: Every preprocessing function should have the following signature:

```python
def preprocessing(frame, config: dict):
```

The 1st argument to `preprocessing` should be a Frame object or a list of Frame objects (for models that utilized multiple Frames). 

Frame is a class that abstracts an image, which is defined under `cvinfer.common.Frame`.  

Basically, Frame defines an interface on what kind of values to be expected from an image and what kind of operations that we could perform. 

The 2nd argument to `preprocessing` should be a dictionary that contains the necessary configuration values needed to preprocess the image. 

These configuration values are those specified under the `preprocessing` section of the `model.configuration.json` file mentioned above in the artifacts. 

For example, we could specify the image resize for resizing in the config dictionary. 

The `preprocessing` function should return 2 values: the 1st value should be the preprocessed inputs that will be passed to the ONNX model and the 2nd value could be any information that will be used later by the postprocessing function. For example, we might want to pass the original image size in this 2nd returned variable in object detection. 

Note that the 1st variable returned by the preprocessing function will be passed to the ONNX model. If your model takes only 1 input, it should be a numpy array of appropriate dtype and shape. If your model expects 2 inputs, it should be a list of 2 numpy arrays. 

**Rule 2**: Every postprocessing function should have the following signature:

```python
def postprocessing(model_output, metadata, config)
```

The 1st argument to `postprocessing` is the output from the ONNX model. Note that if your deep learning model only returns 1 array, `model_output` will be a list of 1 numpy array. 

If your model returns 2 arrays, `model_output` will be a list of 2 numpy arrays. That is, `model_output` is exactly what returned by running the ONNX model. 

The 2nd argument to `postprocessing` is the 2nd variable returned by the preprocessing. This could be any value that we need from the preprocessing step such as the original image size and so on. 

The 3rd argument to `postprocessing` is the configuration dictionary that contains any configuration needed for postprocessing. For example, in object detection, we need confidence threshold and Non-maximum suppression threshold to perform postprocessing. We should access these thresholds via the 3rd argument. The keys should be the same keys that appear under `postprocessing` section in the `configuration.json` file.
