# On learning higher order cumulants in diffusion models


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14035138.svg)](https://doi.org/10.5281/zenodo.14035138)


Data and code release for [arxiv:2410.21212](https://arxiv.org/abs/2410.21212)

## Setting up environment
The code requires:
```
Python >= 3.11 & PyTorch >= 2.0
```

For conda users:

```
conda env create --file=environment.yml
```

and activate using:
```
conda activate score_env
```

## Running the code

Simply running:
```
python bin/train_model.py --file <filename> --max_epochs <max_epochs> --batch_size <batch_size> [--gpu/--no-gpu] [--VE/--no-VE]
```
will run the trainer and save the best trained model's weights. It will also be saving checkpoint every 10 epochs for flexible training.

If you wish to use multiple gpus/cpus and distributed training, you can use `torchrun`, in the following way:

```
torchrun --standalone  --nproc_per_node=<no. of gpus/cpus> bin/train_model.py --file <filename> --max_epochs <max_epochs> --batch_size <batch_size> --ddp [--gpu/--no-gpu] [--VE/--no-VE]
```

Flags:

* [--ddp/--no-ddp] - option whether to use distirbuted data parallelism. Can be used to train on multiple CPUs/GPUs. Recommended when training on multiple or even a single GPU. Doesn't support MPS (Mac).
* [--gpu/--no-gpu] - controls whether to use GPU(s) for training.
* [--VE/--no-VE]   - selects between the variance expanding (VE) or variance preserving (VP) schemes described in the preprint.

Running the notebook `plots.ipynb` will use the trained model as well as other data in `data` folder to generate the aggregated plots and figures found in the paper.

## Field theory code/data

The lattice field theory analysis was done using HMC data and the code found in `https://github.com/Anguswlx/DMasSQ`.
