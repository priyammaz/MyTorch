CUPYX_DISTRIBUTED_HOST=127.0.0.1 CUPYX_DISTRIBUTED_PORT=13333 \
python tests/all_reduce_test_wo_launch.py --rank 0 --world_size 2 &

CUPYX_DISTRIBUTED_HOST=127.0.0.1 CUPYX_DISTRIBUTED_PORT=13333 \
python tests/all_reduce_test_wo_launch.py --rank 1 --world_size 2
