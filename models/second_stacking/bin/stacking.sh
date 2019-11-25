#!/bin/bash

docker run --rm \
    -v $PWD/:/root/ \
    -v $PWD/../../intermediate_output/:/root/intermediate_output/ \
    -v $HOME/.cache/:/root/.cache \
    --runtime=nvidia \
    --ipc=host \
    kaggle/rsna \
    sh bin/stacking_cmd.sh