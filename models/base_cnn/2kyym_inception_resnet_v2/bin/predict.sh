model=model
modelname=2kyym_inception_resnet_v2
gpu=0
ep=2
tta=5
clip=1e-6
conf=./conf/${model}.py

for fold in 0 1 2 3 4
do
snapshot=./intermediate_output/${modelname}/fold${fold}_ep${ep}.pt
valid=./intermediate_output/${modelname}/fold${fold}_valid.pkl
test=./intermediate_output/${modelname}/fold${fold}_test.pkl
docker run --rm \
    -v $PWD/:/root/ \
    -v $PWD/../../../intermediate_output/:/root/intermediate_output/ \
    -v $PWD/../../../input/:/root/input/ \
    -v $HOME/.cache/:/root/.cache \
    --runtime=nvidia \
    --ipc=host \
    kaggle/rsna \
    python -m src.cnn.main test ${conf} --snapshot ${snapshot} --output ${test} --n-tta ${tta} --fold ${fold} --gpu ${gpu}
docker run --rm \
    -v $PWD/:/root/ \
    -v $PWD/../../../intermediate_output/:/root/intermediate_output/ \
    -v $PWD/../../../input/:/root/input/ \
    -v $HOME/.cache/:/root/.cache \
    --runtime=nvidia \
    --ipc=host \
    kaggle/rsna \
    python -m src.cnn.main valid ${conf} --snapshot ${snapshot} --output ${valid} --n-tta ${tta} --fold ${fold} --gpu ${gpu}
done
