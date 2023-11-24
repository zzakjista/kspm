import os
import json
import random
import numpy as np
import torch

def setup_service(args):
    random_seed(args.seed)
    if not args.pretrained:
        experiment_path = create_experiment_folder(args) # create experiment folder
        export_experiments_config_as_json(experiment_path, args) # export experiment config as json
        return experiment_path
    else:
        experiment_path = os.path.join('experiment', args.version, 'pretrained')
        experiment_index = _get_experiment_index(experiment_path)
        experiment_path = experiment_path + '.' +  str(experiment_index)
        os.mkdir(experiment_path)
        return experiment_path
    
def create_experiment_folder(args):
    experiment_root = 'experiment'
    if not os.path.exists(experiment_root):
        os.mkdir(experiment_root)
    experiment_path = get_name_experiment_folder(experiment_root, args)
    os.mkdir(experiment_path)
    print('created experiment folder: ' + os.path.abspath(experiment_path))
    return experiment_path

def get_name_experiment_folder(experiment_root, args):
    if args.model_name == 'VALSTM':
        version = 'v1'
        experiment_path = os.path.join(experiment_root, version)
    elif args.model_name == 'CNNLSTM':
        version = 'v2'
        experiment_path = os.path.join(experiment_root, version)
    experiment_index = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + '.' + str(experiment_index)
    return experiment_path

def _get_experiment_index(experiment_path):
    experiment_index = 0
    while True:
        experiment_name = experiment_path + '.' + str(experiment_index)
        if not os.path.exists(experiment_name):
            break
        experiment_index += 1
    return experiment_index

def export_experiments_config_as_json(experiment_path, args):
    with open(os.path.join(experiment_path, 'params.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=2)

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False