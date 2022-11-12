# Hierarchical Graph Contrastive Learning

This repository is the official implementation of Hierarchical Graph Contrastive Learning.

## Requirements

To install requirements:
For the relevant environment configuration, please see the requirements.txt file

## Training

To train the model(s) in the paper, run this command:

```train
python newmain.py --HGCL_layer 3 --batch_size 64 --far 3 --seed 0 --pooling_ratio 0.9 --trade_off 0.5
```



