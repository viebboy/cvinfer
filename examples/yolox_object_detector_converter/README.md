# Object Detector Converter Demo

This demonstrates the workflow and uses of this interface when generating artifacts for model deployment.

The example takes YOLOX object detection model as an example. 

We could find the preconverted ONNX model in [here](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime).

The inference logics using the ONNX model can be found in [here](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/onnx_inference.py)

Here we should note that the whole inference logics do not just depend on the deep learning model (computational graph) encapsulated in the ONNX format but also preprocessing and postprocessing steps. 

For example, for object detection or image inference in general, we often need to resize an image to a target shape before feeding it to the deep learning model. 

This is a preprocessing step. Similarly, this resized image might also need to be normalized or standardized. 

Popular postprocessing steps in object detection are confidence thresholding and non-maximum suppression.

For this reason, the model developer needs to generate the following artifacts as mentioned in the README.md of the library:

- model.onnx
- model.preprocessing
- model.postprocessing
- model.requirements.txt
- model.configurations.json


## Preprocessing and Postprocessing Artifacts
As a rule of thumb, we want the preprocessing code to be as minimum as possible. Same principle is applied to dependencies. 

If we only use a small part of the logic from a non-standard library for preprocessing or postprocessing, we should reimplement that part of the logic using standard library like numpy and opencv rather than incurring another dependency. 

More dependencies mean less compatibility. 

All the dependencies should be specified in `model.requirements.txt`. 

Note that input to the preprocessing function is always a (or a list of) `common.Frame` object and a configuration dictionary.

For postprocessing, the function should always receive 3 arguments:

- the output from running onnx session
- the metadata returned from the preprocessing function
- the config dictionary used in postprocessing such as confidence threshold
