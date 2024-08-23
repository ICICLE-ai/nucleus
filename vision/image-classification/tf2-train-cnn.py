#!/usr/bin/env python3
#
# Train a simple convolutional neural network to perform image classification

import argparse
import sys

import numpy as np
import tensorflow as tf
import onnx
import tf2onnx


def get_command_arguments():
    """ Read input variables and parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Train a simple convolutional neural network to perform image classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument('-d', '--data', type=str, default=None, 
                        help='path to the image directory')

    parser.add_argument('-m', '--model', type=str, default='base', choices=['base', 'defo'], 
                        help='name of the model to be trained')

    parser.add_argument('-n', '--classes', type=int, default=10, 
                        help='number of classes contained within the dataset')

    parser.add_argument('-H', '--height', type=int, default=32, 
                        help='image height')

    parser.add_argument('-w', '--width', type=int, default=32, 
                        help='image width')

    parser.add_argument('-c', '--channels', type=int, default=3, choices=['1','3','4'], 
                        help='number of color channels')

    parser.add_argument('-p', '--precision', type=str, default='fp32', choices=['bf16', 'fp16', 'fp32', 'fp64'], 
                        help='floating-point precision of model parameters')

    parser.add_argument('-e', '--epochs', type=int, default=42, 
                        help='number of training epochs')

    parser.add_argument('-b', '--batch_size', type=int, default=256, 
                        help='batch size')

    parser.add_argument('-s', '--save_format', type=str, default='keras', choices=['h5','keras', 'onnx', 'tf'], 
                        help='output file format of the trained model')

    parser.add_argument('-v', '--verbose', type=int, default='2', choices=[0, 1, 2], 
                        help='verbosity')

    args = parser.parse_args()
    return args


def create_dataset(data, classes, height, width, channels, dtype):
    """ Create a dataset from a set of images """

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=data,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=None,
        image_size=(height, width),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False)

    # Enforce that the datatypes are the same used in the datasets created by tf.keras.datasets
    dataset = dataset.map(lambda x, y: (tf.cast(x, dtype), tf.cast(y, tf.uint8)))

    return dataset


def create_model(model, classes, height, width, channels):
    """ Specify and compile a convolutional neural network model """

    if model == 'defo':

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(height, width, channels)),
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2=0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2=0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2=0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2=0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2=0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2=0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2=0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', 
                                   kernel_regularizer=tf.keras.regularizers.L2(l2=0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu', 
                                  kernel_regularizer=tf.keras.regularizers.L2(l2=0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(rate=0.35),
            tf.keras.layers.Dense(classes),],
            name=model)

    else: # model == 'base':

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(height, width, channels)),
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(classes),],
            name=model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],) 

    return model


def main():
    """ Train a simple convolutional neural network to perform image classification """

    # Read input variables and parse command-line arguments
    args = get_command_arguments()

    # Set internal variables from input variables and command-line arguments
    if args.precision == 'bf16':
        precision = tf.bfloat16
    elif args.precision == 'fp16':
        precision = tf.float16
    elif args.precision == 'fp64':
        precision = tf.float64
    else: # args.precision == 'fp32'
        precision = tf.float32
    
    # Create training and test datasets
    train = create_dataset(
        data = args.data+'/train', 
        classes = args.classes, 
        height = args.height, 
        width = args.width, 
        channels = args.channels, 
        dtype = precision)

    test = create_dataset(
        data=args.data+'/test', 
        classes=args.classes, 
        height=args.height, 
        width=args.width, 
        channels=args.channels, 
        dtype=precision)

    # Prepare the datasets for training and evaluation
    train = train.cache().shuffle(buffer_size=50000, reshuffle_each_iteration=True).batch(args.batch_size)
    test = test.batch(args.batch_size)

    # Create the model
    model = create_model(
        model=args.model, 
        classes=args.classes, 
        height=args.height, 
        width=args.width, 
        channels=args.channels)

    # Print a summary of the network architecture
    model.summary()

    # Train the model on the dataset
    model.fit(x=train, epochs=args.epochs, verbose=args.verbose)

    # Evaluate the model and its accuracy
    model.evaluate(x=test, verbose=args.verbose)

    # Save the model
    if args.save_format != 'onnx':
        model.save(
            filepath=model.name+'.'+args.save_format, 
            overwrite='True', 
            save_format=args.save_format)
    else:
        onnx_model, _ = tf2onnx.convert.from_keras(
            model = model, 
            input_signature = [tf.TensorSpec(shape=(None, args.height, args.width, args.channels), dtype=precision)], 
            opset = 18,
            custom_ops = None,
            custom_op_handlers = None, 
            custom_rewriter = None,
            inputs_as_nchw = None,
            outputs_as_nchw = None, 
            extra_opset = None,
            shape_override = None, 
            target = None, 
            large_model = False, 
            output_path = None)
        onnx.save(proto=onnx_model, f=model.name+'.onnx')

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
