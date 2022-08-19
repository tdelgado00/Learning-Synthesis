This repository contains the implemantation of a Reinforcement Learning approach for the Directed Controller Synthesis problem.

RL agents are heuristics for the On-the-fly DCS algorithm implemented at [MTSA](https://mtsa.dc.uba.ar/).

This is a work in progress at [Lafhis](https://lafhis.dc.uba.ar/), University of Buenos Aires.

## Requirements

Python requirements can be installed with:
```
$ pip install -r requirements.txt
```

A compiled version of MTSA named mtsa.jar should be available in the base directory.
It can be downloaded from drive using this [link](https://drive.google.com/file/d/1YttxbWW0DBWT_DeFI-5GBqct3_WMgkWX/view?usp=sharing)
or directly compiled from the DCSNonBlockingForRL branch in the MTSA [repo](https://bitbucket.org/lnahabedian/mtsa/src/DCSNonBlockingForRL/).

## Usage

The training pipeline can be executed by running
```
$ python src/train.py
```
from the base directory.

Training parameters can be set by modifying the main function of the src/train.py file. 

Then, trained agents and training and testing statistics can be found in the corresponding
folder of the experiments/results directory.