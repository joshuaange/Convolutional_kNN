#!/bin/sh

echo $HOSTNAME

python CIFAR-10/CIFAR-10.py

echo $HOSTNAME

python CIFAR-10/CIFAR-10_with_convolutions.py

echo $HOSTNAME