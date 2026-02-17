# Hospital Queue Optimization using Reinforcement Learning

## Overview
This repository contains a Reinforcement Learning environment for optimizing queue management and appointment scheduling. It includes the simulation environment, policy definitions, and evaluation scripts.

## Installation

1. Clone the repository.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
```

## Generate Data
Data will be generated in `app/data/data_files`.

```bash
python -m app.data.InstanceGenerator
```

## Project Structure
- **app/data**: Contains instance generation logic and configuration.  
- **app/simulation**: Contains the Gym environment (`Env.py`) and policy definitions.  


## Usage

### Train and Run the Model
To train the PPO agent run:

```bash
python -m app.main2
```
## Note
The repository includes a **pre-trained model** ([ppo_2.zip](ppo_2.zip)).  
You can skip the training step and run the evaluation directly to see the performance of our saved model.

### Evaluation
To evaluate the model on 50 instances to compare performance metrics and get final score, run:

```bash
python -m app.evaluate2
```

## Results
- **Train**: When running `app.main2`, the simulation result for the instance is saved in  
  [`app/data/results2/result_0.csv`](app/data/results2/result_0.csv).  
- **Evaluation**: When running `app.evaluate2`, the detailed solution files for each instance are saved in the  
  `./results/tmp2/` directory.