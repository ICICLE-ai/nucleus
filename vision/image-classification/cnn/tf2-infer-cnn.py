#!/usr/bin/env python3
#
# Perform image classification using a simple convolutional neural network

import argparse
import sys

import numpy as np
import tensorflow as tf
import onnx


def get_command_arguments():
    """ Read input variables and parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Perform image classification using a simple convolutional neural network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument('-d', '--data', type=str, default=None, 
                        help='path to an image')

    parser.add_argument('-m', '--model', type=str, default=None, 
                        help='path to a model')

    parser.add_argument('-v', '--verbose', type=int, default='2', choices=[0, 1, 2], 
                        help='verbosity')

    args = parser.parse_args()
    return args


def main():
    """ Perform image classification using a simple convolutional neural network """

    # Read input variables and parse command-line arguments
    args = get_command_arguments()

    # Load the model
    model = tf.keras.models.load_model(
        filepath=args.model, 
        custom_objects=None, 
        compile=True, 
        safe_mode=True)

    # Print a summary of the network architecture
    model.summary()

    # Read an image from a file and preprocess it
    image = tf.keras.utils.load_img(
        path=args.data, 
        color_mode='rgb', 
        target_size=None, 
        interpolation='nearest', 
        keep_aspect_ratio=False) 

    image = tf.keras.utils.img_to_array(
        img=image, 
        data_format=None, 
        dtype=None)

    image = tf.expand_dims(input=image, axis=0, name=None)

    # Run inference on the image and obtain a prediction from the model
    prediction = model.predict(
        x=image, 
        batch_size=1, 
        verbose='auto', 
        steps=None, 
        callbacks=None)

    # Score the prediction
    score = tf.nn.softmax(logits=prediction[0], axis=None, name=None)

    # Print the result
    print(np.argmax(score), 100*np.max(score))

    return 0


if __name__ == '__main__':
    sys.exit(main())


# References:
# https://www.tensorflow.org/tutorials/images/cnn
# https://touren.github.io/2016/05/31/Image-Classification-CIFAR10.html
# https://towardsdatascience.com/deep-learning-with-cifar-10-image-classification-64ab92110d79
# https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
# https://en.wikipedia.org/wiki/8-bit_color
# https://www.tensorflow.org/guide/keras/sequential_model
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#summary
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile
# https://www.tensorflow.org/guide/keras/train_and_evaluate
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit
# https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#evaluate
# https://www.tensorflow.org/tutorials/load_data/images
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
# https://www.tensorflow.org/tutorials/load_data/tfrecord
# https://onnxruntime.ai/docs/tutorials/tf-get-started.html
# https://github.com/onnx/tensorflow-onnx/blob/main/tutorials/keras-resnet50.ipynb
# https://www.tensorflow.org/tutorials/keras/save_and_load
# https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model
# https://www.tensorflow.org/tutorials/images/classification
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/load_img
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
# https://stackoverflow.com/questions/74368086/tensorflow-inference-on-a-single-image-image-dataset-from-directory
# https://stackoverflow.com/questions/17394882/how-can-i-add-new-dimensions-to-a-numpy-array
