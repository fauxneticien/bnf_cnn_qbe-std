import logging, os, sys, yaml

import torch
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import accuracy_score

from Models import ConvNet
from Datasets import STD_Dataset

def load_parameters(yaml_path):
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def load_std_datasets(dataset):
    return STD_Dataset(dataset['root_dir'], dataset['labels_csv'], dataset['query_dir'], dataset['audio_dir'])

def setup_exp(config):
    output_dir = config['artifacts']['dir']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

def train_model(config):
    output_dir = config['artifacts']['dir']

    setup_exp(config)

    logging.info(' Starting experiment %s' % (config['exp_id']))

    logging.info(' Loading in data from %s' % (config['dataset']['root_dir']))

    dataloader = DataLoader(
        load_std_datasets(config['dataset']),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )

    logging.info(" Instantiating model '%s'" % (config['model_name']))

    model, optimizer, criterion = instantiate_model(config)

    for epoch in range(config['num_epochs']):

        for i_batch, sample_batched in enumerate(dataloader):
            dists = sample_batched['dists']
            labels = sample_batched['labels']

            if (config['use_gpu']):
                dists, labels = dists.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(dists)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            accuracy = accuracy_score(np.round(outputs.cpu().detach()), labels.cpu())

            if(i_batch % 100 == 0):
                logging.info(' Epoch: [%d/%d], Batch: %d, Loss: %.4f, Accuracy: %.2f' % (epoch+1, config['num_epochs'], i_batch+1, loss.data, accuracy))

    logging.info(" Saving model to '%s/model.pt'" % (output_dir))

    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))

    return model
