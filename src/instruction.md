This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:

```
$ python train.py --batch_size=32 --compile=False
```

To run with DDP on 4 gpus on 1 node, example:

```
$ torchrun --standalone --nproc_per_node=4 train.py
```

To run with DDP on 4 gpus across 2 nodes, example:

Run on the first (master) node with example IP `123.456.123.456`:

```
  $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
```

Run on the worker node:

```
  $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

(If your cluster does not have Infiniband interconnect prepend `NCCL_IB_DISABLE=1`)

_source: [https://www.youtube.com/@AndrejKarpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1s)_
