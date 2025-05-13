#!/bin/bash

#Counting the .mat files 
ls /MST-plus-plus/test_develop_code/exp/*.mat | wc -l

#List out the .mat files 
find /MST-plus-plus/test_develop_code/exp/ -type f -name "*.mat"

