This repository contains the implemantation of the learning algorithm for https://arxiv.org/abs/2210.05393.

RL agents are heuristics for the On-the-fly DCS algorithm implemented at [MTSA](https://mtsa.dc.uba.ar/).

This is a work in progress at [Lafhis](https://lafhis.dc.uba.ar/), University of Buenos Aires.

## Installation

  1. Run the following command to install the required python modules.
 ``` 
  python -m pip install -r requirements.txt
 ```
  2. Download the mtsa.jar file from [here](https://drive.google.com/file/d/1YttxbWW0DBWT_DeFI-5GBqct3_WMgkWX/view) and move it to the directory of this readme.
 
## Running experiments

The experiments from the paper can be run with the following command, executing it **from the repo's root directory**.
```console
python src/main.py --exp-path "my experiment"
``` 
All results are then stored at ```experiments/results/my experiment```.