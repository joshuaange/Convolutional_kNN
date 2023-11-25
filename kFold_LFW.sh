#!/bin/sh

echo $HOSTNAME

python LFW/kFoldCV_LFW.py

echo $HOSTNAME

python LFW/kFoldCV_LFW_with_convolutions.py

echo $HOSTNAME