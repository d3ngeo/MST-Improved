#!/bin/bash

#I want to extract file from my google drive

FILE_ID="1A5309Gk7kNFI-ORyADueiPOCMQNTA7r5"
FILE_NAME="Test_RGB.zip"

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILE_ID -O $FILE_NAME

unzip Test_RGB.zip
echo "Done unzipping"
