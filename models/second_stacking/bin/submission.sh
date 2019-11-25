#!/bin/bash
test=./intermediate_output/stacking2nd_lgbm/pred_test.pkl
sub=./output/submission.csv
clip=1e-6

docker run --rm \
    -v $PWD/:/root/ \
    -v $PWD/../../intermediate_output/:/root/intermediate_output/ \
    -v $PWD/../../input/:/root/input/ \
    -v $PWD/../../output/:/root/output/ \
    -v $HOME/.cache/:/root/.cache \
    --runtime=nvidia \
    --ipc=host \
    kaggle/rsna \
    python -m make_submission --input ${test} --output ${sub} --clip ${clip}
