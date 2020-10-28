import logging, os, sys, yaml

import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from tqdm import tqdm

from Models import ConvNet
from Datasets import STD_Dataset

def load_parameters(yaml_path):
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def make_results_csv(csv_path):
    t_df = pd.DataFrame(columns=['epoch', 'query','reference','label','pred'])
    t_df.to_csv(csv_path, index = False)

def append_results_csv(csv_path, results_dict):
    df = pd.DataFrame(results_dict)
    df.to_csv(csv_path, mode = 'a', header = False, index = False)

def load_std_datasets(datasets):
    return { ds_name:STD_Dataset(ds_attrs['root_dir'], ds_attrs['labels_csv'], ds_attrs['query_dir'], ds_attrs['audio_dir']) for (ds_name, ds_attrs) in datasets.items() }

def setup_exp(config):
    output_dir = config['artifacts']['dir']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

def instantiate_model(config):
    constructor = globals()[config['model_name']]
    model = constructor()

    if('model_start' in config.keys()):
        logging.info(" Loading model from '%s'" % (config['model_start']))

        model.load_state_dict(torch.load(config['model_start']))

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

def load_saved_model(pt_path, model_name, mode = 'eval', use_gpu = False):

    model, _, _ = instantiate_model(model_name, mode, config = {"use_gpu": use_gpu})
    model.load_state_dict(torch.load(pt_path))
    
    return model

def save_model(epoch, model, optimizer, loss, output_dir, name = 'model.pt'):
    logging.info(" Saving model to '%s/model-e%d.pt'" % (output_dir, epoch))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, os.path.join(output_dir, 'model.pt'))

def train_model(config):
    output_dir = config['artifacts']['dir']

    setup_exp(config)

    logging.info(' Starting experiment %s' % (config['exp_id']))

    logging.info(' Loading in data from %s' % (config['datasets']['train']['root_dir']))

    datasets = load_std_datasets(config['datasets'])

    train_dat_loader = DataLoader(
        dataset = datasets['train'],
        batch_size = config['datasets']['train']['batch_size'],
        shuffle = True,
        num_workers=config['dl_num_workers']
    )

    if('dev' in datasets.keys()):

        make_results_csv(os.path.join(output_dir, 'dev_results.csv'))

        dev_dat_loader = DataLoader(
            dataset = datasets['dev'],
            batch_size = config['datasets']['dev']['batch_size'],
            shuffle = False,
            num_workers=config['dl_num_workers']
        )

    else:
        # If no dev set defined, then no need to evaluate on dev set every nth epoch
        config['eval_dev_epoch'] = None

    logging.info(" Instantiating model '%s'" % (config['model_name']))

    model, optimizer, criterion = instantiate_model(config)

    for epoch in range(1, config['num_epochs'] + 1):

        epoch_loss = 0

        for i_batch, sample_batched in enumerate(tqdm(train_dat_loader)):
            dists = sample_batched['dists']
            labels = sample_batched['labels']

            if (config['use_gpu']):
                dists, labels = dists.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(dists)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data

            append_results_csv(os.path.join(output_dir, 'train_results.csv'),
            {
                'epoch' : [epoch] * len(sample_batched['query']),
                'query' : sample_batched['query'],
                'reference' : sample_batched['reference'],
                'label' : sample_batched['labels'].reshape(-1).numpy().astype(int),
                'pred' : outputs.cpu().detach().reshape(-1).numpy().round(10)
            })

        logging.info(' Epoch: [%d/%d], Loss: %.4f' % (epoch, config['num_epochs'], epoch_loss / (i_batch + 1)))

        # Evaluate on dev set if not first epoch and current epoch divisible by eval_dev_epoch
        if(config['eval_dev_epoch'] is not None and epoch > 1 and epoch % config['eval_dev_epoch'] == 0):

            with torch.no_grad():

                dev_loss = 0

                for j_batch, dev_batched in enumerate(tqdm(dev_dat_loader)):
                    dists = dev_batched['dists']
                    labels = dev_batched['labels']

                    if (config['use_gpu']):
                        dists, labels = dists.cuda(), labels.cuda()

                    model.eval()
                    outputs = model(dists)

                    dev_loss += criterion(outputs, labels).cpu().data

                    append_results_csv(os.path.join(output_dir, 'dev_results.csv'),
                    {
                        'epoch' : [epoch] * len(dev_batched['query']),
                        'query' : dev_batched['query'],
                        'reference' : dev_batched['reference'],
                        'label' : dev_batched['labels'].reshape(-1).numpy().astype(int),
                        'pred' : outputs.cpu().detach().reshape(-1).numpy().round(10)
                    })

                logging.info(' Epoch: [%d/%d], Dev Loss: %.4f' % (epoch, config['num_epochs'], dev_loss / (j_batch + 1)))

            model.train()

        save_model(epoch, model, optimizer, loss, output_dir)

    return model
