VENV_PATH="venv/bin/activate"

if [ -d $VENV_PATH ] && echo $VENV_PATH
then
    source $VENV_PATH
fi

python ./run.py \
    --lr 0.004 \
    --batch_size 100 \
    --epochs 1 \
    --momentum 0.9 \
    --max_grad_norm 1.0 \
    --weights_path ./saved_weights/mnist_LeNet_private.pth \
    --version private