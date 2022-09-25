# Assignment 3 - Experiment Tracking

<div align="center">

# Data Version Control and Experiment Tracking

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

What it does

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```


### Cifar10 Optuna Sweeper:`cifar_optuna.yaml`
```bash
# @package _global_

# example hyperparameter optimization of some experiment with Optuna:
# python train.py -m hparams_search=mnist_optuna experiment=example

defaults:
  - override /hydra/sweeper: optuna

# choose metric which will be optimized by Optuna
# make sure this is the correct name of some metric logged in lightning module!
optimized_metric: "val/acc_best"

# here we define Optuna hyperparameter search
# it optimizes for value returned from function with @hydra.main decorator
# docs: https://hydra.cc/docs/next/plugins/optuna_sweeper
hydra:
  mode: "MULTIRUN" # set hydra to multirun by default if this config is attached

  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    # storage URL to persist optimization results
    # for example, you can use SQLite if you set 'sqlite:///example.db'
    storage: null

    # name of the study to persist optimization results
    study_name: null

    # number of parallel workers
    n_jobs: 1

    # 'minimize' or 'maximize' the objective
    direction: maximize

    # total number of runs that will be executed
    n_trials: 10

    # choose Optuna hyperparameter sampler
    # you can choose bayesian sampler (tpe), random search (without optimization), grid sampler, and others
    # docs: https://optuna.readthedocs.io/en/stable/reference/samplers.html
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 1234
      n_startup_trials: 10 # number of random sampling runs before optimization starts

    # define hyperparameter search space
    params:
      model.optimizer._target_: choice(torch.optim.Adam, torch.optim.SGD, torch.optim.RMSProp)
      model.optimizer.lr: interval(0.0001, 0.1)
      datamodule.batch_size: choice(8, 16, 24)
```
#### Train.yaml
```bash
# @package _global_

# specify here default configuration
# order of defaults determines the order in which configs override each other
defaults:
  - _self_
  - datamodule: cifar.yaml
  - model: timm.yaml
  - callbacks: default.yaml
  - logger: tensorboard # set logger here or use command line (e.g. `python train.py logger=tensorboard`)
  - trainer: default.yaml
  - paths: default.yaml
  - extras: default.yaml
  - hydra: default.yaml

  # experiment configs allow for version control of specific hyperparameter
  # e.g. best hyperparameters for given model and datamodule
  - experiment: null

  # config for hyperparameter optimization
  - hparams_search: null

  # optional local config for machine/user specific settings
  # it's optional since it doesn't need to exist and is excluded from version control
  - optional local: default.yaml

  # debugging config (enable through command line, e.g. `python train.py debug=default)
  - debug: null

# task name, determines output directory path
task_name: "train"

# tags to help you identify your experiments
# you can overwrite this in experiment configs
# overwrite from command line with `python train.py tags="[first_tag, second_tag]"`
# appending lists from command line is currently not supported :(
# https://github.com/facebookresearch/hydra/issues/1547
tags: ["dev"]

# set False to skip model training
train: True

# evaluate on test set, using best model weights achieved during training
# lightning chooses best weights based on the metric specified in checkpoint callback
test: True

# simply provide checkpoint path to resume training
ckpt_path: null

# seed for random number generators in pytorch, numpy and python.random
seed: null

```
#### Timm.yaml : Which is the configuration of timm model to train on Cifar10 dataset
```bash
_target_: src.models.timm_module.TIMMLitModule

model_name: resnet18

optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001
  weight_decay: 0.0

```
### Train the hyperparameter optuna sweeper
`!python src/train.py  -m trainer=gpu hparams_search=cifar_optuna`

This train the hyperparameter search by optuna on gpu as device.

`!python src/train.py  -m trainer=gpu hparams_search=cifar_optuna model_name=resnet18`

We can pass any timm model name where hyperparameter search takes place by overriding `model_name` argument.


### Push model, logs and data to google drive (using dvc)
1. Untrack logs from git : `git rm -r --cached logs`
2. Add logs to dvc: `dvc add logs`
3. Add a remote: `dvc remote add gdrive gdrive://gdrive_url`
4. Push logs and other tracked files by dvc in gdrive: `dvc push -r gdrive`
5. Now, whenever logs is deleted then, we can directly pull logs from dvc as: `dvc pull -r gdrive`

### Tensorboard 
`tensorboard --logdir logs/train --bind_all`