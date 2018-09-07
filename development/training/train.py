import json
import os
import importlib
import argparse

''''
wandb
''''
import wandb

DEFAULT_TRAINING_ARGS = {
    'batch_size': 64,
    'epochs': 8
}

def run_experiment(experiment_config: Dict, save_weights: bool, use_wandb = False):
    """
    experiment_config is of the form
    {
        "dataset": "EmnistLinesDataset",
        "dataset_args": {
            "max_overlap": 0.4
        },
        "model": "LineModel",
        "algorithm": "line_cnn_sliding_window",
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

    print(f'Running experiment with config {experiment_config}')

    datasets_module = importlib.import_module('development.datasets')
    dataset_class = getattr(datasets_module, experiment_config['dataset'])
    dataset_args = experiment_config.get('dataset_args', {})
    dataset = dataset_class(**dataset_args)
    dataset.load_or_generate_data()
    print(dataset)

    models_module = importlib.import_module('development.models')
    model_class = getattr(models_module, experiment_config['model'])

    algorithm_module = importlib.import_module('development.algorithms')
    algorithm_fn = getattr(algorithm_module, experiment_config['algorithm'])
    algorithm_args = experiment_config.get('algorithm_args', {})
    model = model_class(dataset_cls=dataset_class, algorithm_fn=algorithm_fn, dataset_args=dataset_args, algorithm_args=algorithm_args)
    print(model)

    experiment_config['train_args'] = {**DEFAULT_TRAINING_ARGS, **experiment_config('train_args', {})}
    experiment_config['experiment_group'] = experiment_config.get('experiment_group', None)
    ''''
    Config GPU
    experiment_config['gpu_ind'] = gpu_ind
    ''''


if __name__ == 'main':
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

''''
GPU manager - see line 115 in https://github.com/julioadl/fsdl-text-recognizer-project/blob/master/lab5_sln/training/run_experiment.py
''''

    experiment_config = json.loads(args.experiment_config)
#    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    run_experiment(experiment_config, args.save, args.gpu)
