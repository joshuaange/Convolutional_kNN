#!/bin/sh

echo $HOSTNAME

python MNIST/kFoldCV_Fashion_MNIST.py

echo $HOSTNAME

python MNIST/kFoldCV_Fashion_MNIST_with_convolutions.py

echo $HOSTNAME