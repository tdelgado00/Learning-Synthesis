This repository contains the implemantation of the learning algorithm for https://arxiv.org/abs/2210.05393.

RL agents are heuristics for the On-the-fly DCS algorithm implemented at [MTSA](https://mtsa.dc.uba.ar/).

This is a work in progress at [Lafhis](https://lafhis.dc.uba.ar/), University of Buenos Aires.

## Installation

  1. Run the following command to install the required python modules.
 ``` 
  python -m pip install -r requirements.txt
 ```
  2. Move the mtsa.jar file to the directory of this readme.
 
## Running experiments

### Training and testing an agent
You can fully reproduce variations of the paper's experimental pipeline by using the following functions:
1. ```train_agent``` to train the agent in a certain context
2. ```test_training_agents_generalization``` to test a random subset of the agent's copies during training.
3. ```test_agent_all_instances``` to perform the selection step across the agents tested in 2 and testing it across multiple contexts.

The exact experiments from the paper can be run with the following command, executing it **from the repo's root directory**.
```console
python3 src/train.py <agent_directory_name> 
``` 
The trained agents are then stored at ```experiments/results/<used_training_context>/<agent_directory_name>``` in onnx format with a respective JSON file containing all of the training's hyperparameters information.

The  results from step 2 are stored in the same directory at the ```generalization_all.csv``` file.

The final results (from step 3) are stored in the same directory as well, but at the ```all_<problem>_15_-1_TO:10m.csv``` and ```all_<problem>_15_15000_TO:3h.csv```.

### Testing the random and RA heuristic
You can do so by calling the ```test_random_all_instances``` and ```test_ra_all_instances``` functions respectively, or directly running the following command.
```console
python3 src/testing.py 
```
```experiments/results/random``` will store random's results
```experiments/results/```will store RA's results under the names  ```all_ra_15_-1_	10m.csv``` and ```all_ra_15_15000_3h.csv```.
