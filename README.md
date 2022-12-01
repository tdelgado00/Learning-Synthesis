# SUPPLEMENTARY MATERIAL GUIDE

## 1. Directory Layout
  
  The supplemantray material's directory layout is as below.

```  
.
+-- fsp-------------------#The specifications of all benchmark problems (in FSP language)

+-- labels-----------------#Set of event labels of the benchmark problems (needed for feature calculation)
		
+-- src-------------------#The implementation of the learning algorithm and the scripts used in the experiments
```

## 2. Running the experiments

### 2.2. Installation
  
  1. From the directory of this readme, run the following command to install the required python modules.
 ``` 
  python -m pip install -r requirements.txt
 ```
  2. Move the mtsa.jar file the directory of this readme.
   
 
 
### 2.3. Framework usage
You can fully reproduce variations of the paper's experimental pipeline by using the following functions:

#### 2.3.1. Training and testing an agent
Make sure you execute them in order:
1. ```train_agent``` to train the agent in a certain context
2. ```test_training_agents_generalization``` to test a random subset of the agent's copies during training.
3. ```test_agent_all_instances``` to perform the selection step across the agents tested in 2 and testing it across multiple contexts.

#### 2.3.2. Testing the random and RA heuristic
You can do so by calling the ```test_random_all_instances``` and ```test_ra_all_instances``` functions respectively.

### 2.4. Reproducing the exact paper's experiments 
The whole experimental pipeline can be run with the 2 terminal commands described below, maintaining the scripts as they are by default. Make sure you execute them **from the repo's root directory**.
##### Train and test the agent (2.3.1): 
```console
python3 src/train.py <agent_directory_name> 
``` 
The trained agents are then stored at ```experiments/results/<used_training_context>/<agent_directory_name>``` in onnx format with a respective JSON file containing all of the training's hyperparameters information.

The  results from step 2 are stored in the same directory at the ```generalization_all.csv``` file.

The final results (from step 3) are stored in the same directory as well, but at the ```all_<problem>_15_-1_TO:10m.csv``` and ```all_<problem>_15_15000_TO:3h.csv```.


##### Test random and RA (2.3.2):
```console
python3 src/testing.py 
```
```experiments/results/random``` will store random's results
```experiments/results/```will store RA's results under the names  ```all_ra_15_-1_	10m.csv``` and ```all_ra_15_15000_3h.csv```.

### 2.5. Functions main parameters description


1. ```train_agent(instances, file, agent_params, features, total_steps=500000, copy_freq=5000, early_stopping=True)```
 
+ *agent_params:* a dictionary with the agent's hyperparameter values. See the original example for better understanding.
+ *features:* a dictionary with the truth value per feature name, for indicating which ones to use. See the original example for better understanding.
+ *total_steps:* an integer indicating the max number of training expansions per agent
+  *copy_freq:* the total number of training steps between each saved version of the agent.
+ *early_stopping (boolean):* if True, it establishes a criterion for stopping training when no improvements are shown after significant amounts of steps.

2. ```test_training_agents_generalization(problem, file, up_to, timeout, total=100, max_frontier=1000000,  ```
  ```solved_crit=budget_and_time, ebudget = -1)```


+ *problem:* a string indicating the name of the problem to test the agent at.
+ *file:* indicates the name of the directory where the agents are to be stored at the ```experiments/results/<used_training_context>``` directory.
+ *up_to* indicates the maximum value used for the $n,k$ possible combinations  of the specified problem to test the specified agent at.
+ *timeout:* a string indicating the time budget allowed for the agent to solve one single context
+ *total:* the size of the used subset.
+ *max_frontier* the budget of possible actions allowed in a single expansion step.
+ *solved_crit* a function that decides whether or not continue testing when the previous adjacent context were not solved (by default, when the adjacent contexts exceeded the expansion and/or time budget).
+ *ebudget*: an integer that specifies the expansions budget. -1 indicates no limit.

**Note**: *test_ra_all_instances* and *test_random_all_instances* use the same parameters but without the agent path specification (file).

3. ```test_agent_all_instances(problem, file, up_to, timeout="10m", name="all", selection=best_generalization_agent_ebudget,```
``` max_frontier=1000000,  fast_stop=True, ebudget=-1, solved_crit=budget_and_time)```
+ *name*: a prefix for the resulting csv name.
+ *selection*: specifies the criterion for selecting the best agent.
+ *fast_stop*: if True, stops evaluation when one previous adjacent context was not solved. if False, it stops evaluation when two of the previous adjacent context were not solved.
+ *solved_crit*: the criterion used to evaluate if a context was solved or not. 
The rest have the same as meaning as in 2.

