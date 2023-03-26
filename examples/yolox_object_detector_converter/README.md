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
- processor.py
- requirements.txt
- configurations.json

The script `detect_object.py` demonstrates how to use the artifacts using the abstractions provided in `cvinfer`.  
