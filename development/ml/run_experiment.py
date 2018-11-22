import json
import os
import importlib
import argparse
from typing import Dict

from training.train import train_model

'''
wandb
import wandb
'''

DEFAULT_TRAINING_ARGS = {
    'batch_size': 64,
    'epochs': 8
}

def run_experiment(experiment_config: Dict, save_weights: bool, gpu_ind:int, use_wandb = False):

    """
    experiment_config is of the form
    {
        "dataset": "sklearnDigits",
        "dataset_args": {
            "max_overlap": 0.4
        },
        "model": "SVMModel",
        "model_backend": "SKLearn",
        "algorithm": "SVM",
        "algorithm_args": {
            "window_width": 14,
            "window_stride": 7
        },
        "train_args": {
            "batch_size": 128,
            "epochs": 10
        }
    }
    save_weights: if True, will save the final model weights to a canonical location (see Model in models/base.py)
    gpu_ind: integer specifying which gpu to use
    """

    print(f'Running experiment with config {experiment_config} on GPU {gpu_ind}')

    datasets_module = importlib.import_module('datasets')
    dataset_cls = getattr(datasets_module, experiment_config['dataset'])
    dataset_args = experiment_config.get('dataset_args', {})
    dataset = dataset_cls(**dataset_args)
    dataset.load_or_generate_data()
    print(dataset)

    models_module = importlib.import_module('models')
    model_cls = getattr(models_module, experiment_config['model'])

    algorithm_module = importlib.import_module('algorithms')
    algorithm_fn = getattr(algorithm_module, experiment_config['algorithm'])
    algorithm_args = experiment_config.get('algorithm_args', {})
    model = model_cls(dataset_cls=dataset_cls, algorithm_fn=algorithm_fn, dataset_args=dataset_args, algorithm_args=algorithm_args)
    print(model)

    experiment_config['train_args'] = {**DEFAULT_TRAINING_ARGS, **experiment_config.get('train_args', {})}
    experiment_config['experiment_group'] = experiment_config.get('experiment_group', None)

    #Config GPU
    experiment_config['gpu_ind'] = gpu_ind


    #if use_wandb:
    #   wandb.init()
    #   wandb.config.update(experiment_config)

    train_model(
        model,
        dataset,
        epochs = experiment_config['train_args']['epochs'],
        batch_size = experiment_config['train_args']['batch_size'],
        gpu_ind = gpu_ind
        #use_wandb = use_wandb
    )

    score = model.evaluate(dataset.x_test, dataset.y_test)
    print(f'Test evaluation:\n s {score}')

    #if use_wandb:
    #   wandb.log({'test_metric': score})

    if save_weights:
        model.save_weights()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Provide index of GPU to use"
    )
    parser.add_argument(
        "--save",
        default=False,
        dest='save',
        action='store_true',
        help='If true, final weights are saved to verion-controlled location'
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help="JSON of config for experiment"
    )
    args = parser.parse_args()

    '''
    GPU manager - see line 115 in https://github.com/julioadl/fsdl-text-recognizer-project/blob/master/lab5_sln/training/run_experiment.py
    '''
    experiment_config = json.loads(args.experiment_config)

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_experiment(experiment_config, args.save, args.gpu, False)
