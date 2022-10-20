"""
########################################################################
Runs a combination of hyperparameters or performs hyperparameter tuning
for the given model, feature mode, and data directory.
########################################################################
"""

import torch
import os
from src.prepare_data import prepare_data
from src.test import test
from src.hyperparameter_tuning import hyperparameter_tuning
from src.train import traindata
from utils import print_metrics, seed_everything, read_config

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed_everything()

def perform(config):
    if config['do_tuning']:
        hyperparameter_tuning(device, config)
    else:
        model_label = config['combination']['model_label']
        mode = config['combination']['mode']
        data_dir = config['combination']['data_dir']
        class_weights = config['combination']['class_weights']
        batch_size = config['hyperparameters']['batch_size']
        kmer_one_hot = config['fixed_vals']['kmer_one_hot']
        hyperparameters = config['hyperparameters']
        
        # train the model
        model = traindata(hyperparameters, device, config)
        
        # get the test data
        _, testloader, _, _, _ = prepare_data(
            data_dir=data_dir,
            mode=mode,
            batch_size=batch_size,
            k=kmer_one_hot,
        )

        # test the model
        metrics = test(device, model, testloader)
        print_metrics(model_label, mode, data_dir, kmer_one_hot, class_weights, metrics)
        
if __name__ == '__main__':
    config = read_config(device)
    perform(config)
