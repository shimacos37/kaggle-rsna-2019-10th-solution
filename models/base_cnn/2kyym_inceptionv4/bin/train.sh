model=model
gpu=0
fold=4
conf=./conf/${model}.py

for fold in 0 1 2 3 4
do
docker run --rm \
    -v $PWD/:/root/ \
    -v $PWD/../../../intermediate_output/:/root/intermediate_output/ \
    -v $PWD/../../../input/:/root/input/ \
    -v $HOME/.cache/:/root/.cache \
    --runtime=nvidia \
    --ipc=host \
    kaggle/rsna \
    python -m src.cnn.main train ${conf} --fold ${fold} --gpu ${gpu}
done