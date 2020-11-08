import logging, os, sys, yaml

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import pandas as pd
import numpy as np
from tqdm import tqdm

from Models import *
from Datasets import STD_Dataset

def load_parameters(yaml_path):
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def load_std_datasets(datasets, apply_vad):
    return {
        ds_name:STD_Dataset(
            root_dir = ds_attrs['root_dir'],
            labels_csv = ds_attrs['labels_csv'],
            query_dir = ds_attrs['query_dir'],
            audio_dir = ds_attrs['audio_dir'],
            apply_vad = apply_vad
        ) for (ds_name, ds_attrs) in datasets.items()
    }

def create_data_loaders(loaded_datasets, config):
    return {
        ds_name:DataLoader(
            dataset = dataset,
            batch_size = config['datasets'][ds_name]['batch_size'],
            shuffle = True if ds_name == 'train' else False,
            num_workers = config['dl_num_workers']
        ) for (ds_name, dataset) in loaded_datasets.items()
    }

def load_saved_model(config):

    model, optimizer, criterion = instantiate_model(config)

    logging.info(" Loading model from '%s'" % (config['model_path']))
    checkpoint = torch.load(config['model_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if(config['mode'] == 'train'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if config['mode'] == 'eval' and config['use_gpu'] and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model, optimizer, criterion

def setup_exp(config):
    output_dir = config['artifacts']['dir']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if(config['mode'] == 'train'):
        make_results_csv(os.path.join(output_dir, 'train_results.csv'))

    logging.basicConfig(
        filename = os.path.join(output_dir, config['artifacts']['log']),
        level = logging.DEBUG,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    return output_dir

def instantiate_model(config):
    constructor = globals()[config['model_name']]
    model = constructor()

    logging.info(" Instantiating model '%s'" % (config['model_name']))

    if config['use_gpu']:
        model.cuda()

    if(config['mode'] == 'train'):
        model.train()

        if(config['optimizer'] == 'adam'):
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) 

        if(config['criterion'] == 'BCELoss'):
            criterion = torch.nn.BCELoss()

        return model, optimizer, criterion

    if(config['mode'] == 'eval'):
        model.eval()
        
        return model, None, None

def make_results_csv(csv_path, headers = 'train'):
    if (headers == 'train'):
        csv_cols = ['epoch', 'query','reference','label','pred']
    elif (headers == 'eval'):
        csv_cols = ['query','reference','label','pred']

    t_df = pd.DataFrame(columns=csv_cols)
    t_df.to_csv(csv_path, index = False)
    return csv_path

def append_results_csv(csv_path, results_dict):
    df = pd.DataFrame(results_dict)
    df.to_csv(csv_path, mode = 'a', header = False, index = False)

def save_model(epoch, model, optimizer, loss, output_dir, name = 'model.pt'):
    cps_path = os.path.join(output_dir, 'checkpoints')
    cp_name  = "model-e%s.pt" % (str(epoch).zfill(3))

    if not os.path.exists(cps_path):
        os.makedirs(cps_path)

    logging.info(" Saving model to '%s/%s'" % (cps_path, cp_name))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, os.path.join(cps_path, cp_name)
    )

def run_model(model, mode, ds_loader, use_gpu, csv_path, keep_loss, criterion, optimizer, epoch):
    if keep_loss is True:
        total_loss = 0
    else:
        # Set to None for referencing in return
        loss = None

    for batch_index, batch_data in enumerate(tqdm(ds_loader)):
        dists  = batch_data['dists']
        labels = batch_data['labels']

        if (use_gpu):
            dists, labels = dists.cuda(), labels.cuda()

        if(mode == 'train'):
            model.train()
            optimizer.zero_grad()
        
        elif(mode == 'eval'):
            model.eval()

        outputs = model(dists)

        if keep_loss is True:
            loss        = criterion(outputs, labels)
            total_loss += loss.cpu().data

            if(mode == 'train'):
                loss.backward()
                optimizer.step()

        batch_output = {}

        if(epoch is not None):
            batch_output['epoch'] = [epoch] * len(batch_data['query'])

        batch_output['query'] = batch_data['query']
        batch_output['reference'] = batch_data['reference']
        batch_output['label'] = batch_data['labels'].reshape(-1).numpy().astype(int)
        batch_output['pred'] = outputs.cpu().detach().reshape(-1).numpy().round(10)

        append_results_csv(csv_path, batch_output)

    if keep_loss is True:
        mean_loss = total_loss / len(ds_loader)
    else:
        # Set to None for referencing in return
        mean_loss = None

    return model, optimizer, criterion, loss, mean_loss
