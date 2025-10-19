python -m mytorch.distributed.launch \
    --num_gpus 2 \
    --training_script tests/all_reduce_test_w_launch.py