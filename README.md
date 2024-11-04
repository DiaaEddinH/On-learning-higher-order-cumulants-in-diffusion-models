# On learning higher order cumulants in diffusion models

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
python bin/train_model.py --max_epochs <max_epochs> --batch_size <batch_size> [--gpu/--no-gpu] [--VE/--no-VE]
```
will run the trainer and save the best trained model's weights. It will also be saving checkpoint every 10 epochs for flexible training.

If you wish to use multiple gpus/cpus and distributed training, you can use `torchrun`, in the following way:

```
torchrun --standalone  --nproc_per_node=<no. of gpus/cpus> bin/train_model.py --max_epochs <max_epochs> --batch_size <batch_size> --ddp [--gpu/--no-gpu] [--VE/--no-VE]
```

Running the notebook `plots.ipynb` will use the trained model as well as other data in `data` folder to generate the aggregated plots and figures found in the paper.

## Field theory code/data

The lattice field theory analysis was done using HMC data and the code found in `https://github.com/Anguswlx/DMasSQ`.
