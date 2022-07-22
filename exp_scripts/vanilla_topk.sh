VENV_PATH="venv/bin/activate"

if [ -d $VENV_PATH ] && echo $VENV_PATH
then
    source $VENV_PATH
fi

python ./private_compress_exp.py \
    --lr 0.004 \
    --batch_size 60 \
    --epochs 15 \
    --max_grad_norm 1.0 \
    --compress True