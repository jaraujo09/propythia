"""
########################################################################

Runs a combination of hyperparameters or performs hyperparameter tuning
for the given model, feature mode, and data directory.

########################################################################
"""

import torch
from torch import nn
import os
from src.prepare_data import prepare_data
from src.test import test
from src.hyperparameter_tuning import hyperparameter_tuning
from src.train import traindata
from ray import tune
import numpy


numpy.random.seed(2022)
torch.manual_seed(2022)
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def perform(model_label, mode, data_dir, do_tuning):
    if('essential_genes' in data_dir):
        class_weights = torch.tensor([1.0, 2.0]).to(device)
    else:
        class_weights = torch.tensor([1.0, 1.0]).to(device)

    combinations = {
        'mlp': ['descriptor'],
        'net': ['one_hot', 'chemical', 'kmer_one_hot'],
        'cnn': ['one_hot', 'chemical', 'kmer_one_hot'],
        'lstm': ['one_hot', 'chemical', 'kmer_one_hot'],
        'gru': ['one_hot', 'chemical', 'kmer_one_hot'],
        'bi_lstm': ['one_hot', 'chemical', 'kmer_one_hot'],
        'cnn_lstm': ['one_hot', 'chemical', 'kmer_one_hot'],
        'cnn_bi_lstm': ['one_hot', 'chemical', 'kmer_one_hot'],
        'buckle_cnn_lstm': ['one_hot', 'chemical', 'kmer_one_hot'],
        'buckle_cnn_bi_lstm': ['one_hot', 'chemical', 'kmer_one_hot'],
    }
    if(model_label in combinations):
        if(mode not in combinations[model_label]):
            raise ValueError(model_label, 'does not support', mode, ', please choose one of', combinations[model_label])
    else:
        raise ValueError('Model label:', model_label, 'not implemented in', combinations.keys())

    fixed_vals = {
        'epochs': 500,
        'optimizer_label': 'adam',
        'loss_function': nn.CrossEntropyLoss(weight=class_weights),
        'patience': 8,
        'output_size': 2,
        'model_label': model_label,
        'data_dir': data_dir,
        'mode': mode,
        'cpus_per_trial':1, 
        'gpus_per_trial':0,
        'num_samples': 15,
        'num_layers': 2,
        'kmer_one_hot': 3, # k value for the kmer one hot encoding
    }

    if do_tuning:
        config = {
            "hidden_size": tune.choice([32, 64, 128, 256]),
            "lr": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([8, 16, 32]),
            "dropout": tune.uniform(0.3, 0.5)
        }

        hyperparameter_tuning(device, fixed_vals, config)
    else:
        config = {
            "hidden_size": 32,
            "lr": 1e-3,
            "batch_size": 32,
            "dropout": 0.35
        }
        model = traindata(config, device, fixed_vals, is_tuning=False)
        
        _, testloader, _, _, _ = prepare_data(
            data_dir=fixed_vals['data_dir'],
            mode=fixed_vals['mode'],
            batch_size=config['batch_size'],
            k=fixed_vals['kmer_one_hot'],
        )
        
        acc, mcc, report = test(device, model, testloader)
        print('Accuracy: %.3f' % acc)
        print('MCC: %.3f' % mcc)
        print(report)

# --------------------------------- Primer ----------------------------------

# --- Descriptors ---
# perform('mlp', 'descriptor', 'primer')

# --- One hot encoding ---
perform('cnn', 'one_hot', 'primer', do_tuning=False)
# perform('lstm', 'one_hot', 'primer')
# perform('gru', 'one_hot', 'primer')
# perform('bi_lstm', 'one_hot', 'primer')
# perform('cnn_lstm', 'one_hot', 'primer')
# perform('cnn_bi_lstm', 'one_hot', 'primer')
# perform('buckle_cnn_lstm', 'one_hot', 'primer')
# perform('buckle_cnn_bi_lstm', 'one_hot', 'primer')

# --- Chemical encoding ---
# perform('cnn', 'chemical', 'primer')
# perform('lstm', 'chemical', 'primer')
# perform('gru', 'chemical', 'primer')
# perform('bi_lstm', 'chemical', 'primer')
# perform('cnn_lstm', 'chemical', 'primer')
# perform('cnn_bi_lstm', 'chemical', 'primer')
# perform('buckle_cnn_lstm', 'chemical', 'primer')
# perform('buckle_cnn_bi_lstm', 'chemical', 'primer')

# --- Kmer One hot encoding ---
# perform('cnn', 'kmer_one_hot', 'primer')
# perform('lstm', 'kmer_one_hot', 'primer')
# perform('gru', 'kmer_one_hot', 'primer')
# perform('bi_lstm', 'kmer_one_hot', 'primer')
# perform('cnn_lstm', 'kmer_one_hot', 'primer')
# perform('cnn_bi_lstm', 'kmer_one_hot', 'primer')
# perform('buckle_cnn_lstm', 'kmer_one_hot', 'primer')
# perform('buckle_cnn_bi_lstm', 'kmer_one_hot', 'primer')

# ----------------------------- Essential genes -----------------------------

# --- Descriptors ---
# perform('mlp', 'descriptor', 'essential_genes/descriptors_all_small_seqs')
# perform('mlp', 'descriptor', 'essential_genes/descriptors_filtered_20k')
# perform('mlp', 'descriptor', 'essential_genes/descriptors_filtered_50k')
# perform('mlp', 'descriptor', 'essential_genes')

# --- One hot encoding ---
# perform('cnn', 'one_hot', 'essential_genes')
# perform('lstm', 'one_hot', 'essential_genes')
# perform('gru', 'one_hot', 'essential_genes')
# perform('bi_lstm', 'one_hot', 'essential_genes')
# perform('cnn_lstm', 'one_hot', 'essential_genes')
# perform('cnn_bi_lstm', 'one_hot', 'essential_genes')
# perform('buckle_cnn_lstm', 'one_hot', 'essential_genes')
# perform('buckle_cnn_bi_lstm', 'one_hot', 'essential_genes')

# --- Chemical encoding ---
# perform('cnn', 'chemical', 'essential_genes')
# perform('lstm', 'chemical', 'essential_genes')
# perform('gru', 'chemical', 'essential_genes')
# perform('bi_lstm', 'chemical', 'essential_genes')
# perform('cnn_lstm', 'chemical', 'essential_genes')
# perform('cnn_bi_lstm', 'chemical', 'essential_genes')
# perform('buckle_cnn_lstm', 'chemical', 'essential_genes')
# perform('buckle_cnn_bi_lstm', 'chemical', 'essential_genes')

# --- Kmer One hot encoding ---
# perform('cnn', 'kmer_one_hot', 'essential_genes')
# perform('lstm', 'kmer_one_hot', 'essential_genes')
# perform('gru', 'kmer_one_hot', 'essential_genes')
# perform('bi_lstm', 'kmer_one_hot', 'essential_genes')
# perform('cnn_lstm', 'kmer_one_hot', 'essential_genes')
# perform('cnn_bi_lstm', 'kmer_one_hot', 'essential_genes')
# perform('buckle_cnn_lstm', 'kmer_one_hot', 'essential_genes')
# perform('buckle_cnn_bi_lstm', 'kmer_one_hot', 'essential_genes')

# ----------------------------------- H3 ------------------------------------

# --- Descriptors ---
# perform('mlp', 'descriptor', 'h3')

# --- One hot encoding ---
# perform('cnn', 'one_hot', 'h3')
# perform('lstm', 'one_hot', 'h3')
# perform('gru', 'one_hot', 'h3')
# perform('bi_lstm', 'one_hot', 'h3')
# perform('cnn_lstm', 'one_hot', 'h3')
# perform('cnn_bi_lstm', 'one_hot', 'h3')
# perform('buckle_cnn_lstm', 'one_hot', 'h3')
# perform('buckle_cnn_bi_lstm', 'one_hot', 'h3')

# --- Chemical encoding ---
# perform('cnn', 'chemical', 'h3')
# perform('lstm', 'chemical', 'h3')
# perform('gru', 'chemical', 'h3')
# perform('bi_lstm', 'chemical', 'h3')
# perform('cnn_lstm', 'chemical', 'h3')
# perform('cnn_bi_lstm', 'chemical', 'h3')
# perform('buckle_cnn_lstm', 'chemical', 'h3')
# perform('buckle_cnn_bi_lstm', 'chemical', 'h3')

# --- Kmer One hot encoding ---
# perform('cnn', 'kmer_one_hot', 'h3')
# perform('lstm', 'kmer_one_hot', 'h3')
# perform('gru', 'kmer_one_hot', 'h3')
# perform('bi_lstm', 'kmer_one_hot', 'h3')
# perform('cnn_lstm', 'kmer_one_hot', 'h3')
# perform('cnn_bi_lstm', 'kmer_one_hot', 'h3')
# perform('buckle_cnn_lstm', 'kmer_one_hot', 'h3')
# perform('buckle_cnn_bi_lstm', 'kmer_one_hot', 'h3')