python mytorch/distributed/launch.py --num_gpus 2 --training_script tests/train_ddp_mnist.py -- \
    --batch_size 32 \
    --lr 0.001
