#!/bin/sh

echo $HOSTNAME

python MNIST/MNIST.py

echo $HOSTNAME

python MNIST/MNIST_with_convolutions.py

echo $HOSTNAME