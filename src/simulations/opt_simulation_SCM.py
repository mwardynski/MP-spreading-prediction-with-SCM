
import json
import optuna
import os
import sys
import yaml

import simulation_SCM

def objective(trial):
    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_path =  config['paths']['dataset_dir'] +  f'datasets/dataset_{category}.json'
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    num_cores = 10
    I_percentage = 1
    Nsteps = 50

    mu = trial.suggest_float('mu', 0.01, 0.1)
    lambda1 = trial.suggest_float('lambda1', 0.0001, 1.5)
    lambdaD = trial.suggest_float('lambdaD', 0.0001, 2.5)
    
    return simulation_SCM.exec_sim(dataset, simulation_SCM.Results(), num_cores, mu, lambda1, lambdaD, I_percentage, Nsteps)

if __name__=="__main__":
    category = sys.argv[1] if len(sys.argv) > 1 else 'ALL'
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print(f"Best parameters: {study.best_params}")
    print(f"Minimal MSE: {study.best_value}")




