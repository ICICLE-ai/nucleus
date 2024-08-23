#!/usr/bin/env python3
#
# Perform image classification using a simple convolutional neural network

import argparse
import sys

import cv2
import numpy as np
import scipy as sp
import onnxruntime as ort


def get_command_arguments():
    """ Read input variables and parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Perform image classification using a simple convolutional neural network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument('-d', '--data', type=str, default=None, 
                        help='path to the image')

    parser.add_argument('-m', '--model', type=str, default=None, 
                        help='path to the model')

    parser.add_argument('-v', '--verbose', type=int, default='2', choices=[0, 1, 2], 
                        help='verbosity')

    args = parser.parse_args()
    return args


def main():
    """ Perform image classification using a simple convolutional neural network """

    # Read input variables and parse command-line arguments
    args = get_command_arguments()

    # Set options for inference session
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1

    # Load the model and an create inference session
    session = ort.InferenceSession(
        path_or_bytes=args.model, 
        sess_options=options, 
        providers=['CPUExecutionProvider'], 
        provider_options=None)

    # Read an image from a file and preprocess it
    image = cv2.imread(filename=args.data, flags=cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = np.expand_dims(a=image, axis=0)

    # Run inference on the image and obtain a prediction from the model
    prediction = session.run(
        output_names=[session.get_outputs()[0].name], 
        input_feed={session.get_inputs()[0].name: image}, 
        run_options=None)

    # Score the prediction
    score = sp.special.softmax(x=np.array(prediction).squeeze(), axis=None)

    # Print the result
    print(np.argmax(score), 100*np.max(score))

    return 0


if __name__ == '__main__':
    sys.exit(main())


# References:
# https://thenewstack.io/tutorial-using-a-pre-trained-onnx-model-for-inferencing/
# https://github.com/onnx/tensorflow-onnx#cli-reference
# https://github.com/onnx/tensorflow-onnx/blob/main/examples/getting_started.py
# https://onnxruntime.ai/docs/tutorials/tf-get-started.html
# https://github.com/onnx/tensorflow-onnx/blob/main/tutorials/keras-resnet50.ipynb
# https://onnxruntime.ai/docs/get-started/with-python.html
# https://onnxruntime.ai/docs/tutorials/iot-edge/rasp-pi-cv.html
# https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/resnet50_modelzoo_onnxruntime_inference.ipynb
# https://onnxruntime.ai/docs/api/python/api_summary.html#inferencesession
# https://onnxruntime.ai/docs/api/python/api_summary.html#api-overview
# https://onnxruntime.ai/docs/execution-providers
# https://github.com/microsoft/onnxruntime/issues/8313
# https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
# https://stackoverflow.com/questions/57325720/opencv-convert-uint8-image-to-float32-normalized-image
# https://stackoverflow.com/questions/17394882/how-can-i-add-new-dimensions-to-a-numpy-array
# https://stackoverflow.com/questions/58834493/what-is-the-default-color-space-in-pil
# https://stackoverflow.com/questions/48789661/rgb-or-bgr-for-tensorflow-slim-resnet-v2-pre-trained-model
# https://www.geeksforgeeks.org/reading-image-opencv-using-python
