#!/bin/bash

ffmpeg -i output.mp4 -c:v libvpx-vp9 -b:v 3M -pass 1 -row-mt 1 -an -f null /dev/null && \
ffmpeg -i output.mp4 -c:v libvpx-vp9 -b:v 3M -pass 2 -row-mt 1 -an output.webm
