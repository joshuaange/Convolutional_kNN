#!/bin/sh

echo $HOSTNAME

python LFW/LFW.py

echo $HOSTNAME

python LFW/LFW_with_convolutions.py

echo $HOSTNAME