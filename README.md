# On learning higher order cumulants in diffusion models

Data and code release for [arxiv:2410.21212](https://arxiv.org/abs/2410.21212)

## Setting up environment
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
python bin/train_model.py --max_epochs <max_epochs> --batch_size <batch_size> [--ddp/--no-ddp] [--gpu/--no-gpu] [--VE/--no-VE]
```
will run the trainer and save the best trained model's weights.

Running the notebook `plots.ipynb` will use the trained model as well as other data in `data` folder to generate the aggregated plots and figures found in the paper.
