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

def load_std_datasets(datasets):
    return { ds_name:STD_Dataset(ds_attrs['root_dir'], ds_attrs['labels_csv'], ds_attrs['query_dir'], ds_attrs['audio_dir']) for (ds_name, ds_attrs) in datasets.items() }

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

def save_model(epoch, model, optimizer, loss, output_dir, name = 'model.pt'):
    logging.info(" Saving model to '%s/model.pt'" % (output_dir))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, output_dir + 'model.pt')

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
        num_workers=0
    )

    if('dev' in datasets.keys()):

        dev_dat_loader = DataLoader(
            dataset = datasets['dev'],
            batch_size = config['datasets']['dev']['batch_size'],
            shuffle = True,
            num_workers=0
        )

    else:
        # If no dev set defined, then no need to evaluate on dev set every nth epoch
        config['eval_dev_epoch'] = None

    logging.info(" Instantiating model '%s'" % (config['model_name']))

    model, optimizer, criterion = instantiate_model(config)

    for epoch in range(config['num_epochs']):

        for i_batch, sample_batched in enumerate(train_dat_loader):
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
                logging.info(' Epoch: [%d/%d], Train Batch: %d, Train Batch Loss: %.4f, Train Batch Accuracy: %.2f' % (epoch+1, config['num_epochs'], i_batch+1, loss.data, accuracy))

        if(config['eval_dev_epoch'] is not None and epoch % config['eval_dev_epoch'] == 0):

            save_model(epoch, model, optimizer, loss, output_dir)

            with torch.no_grad():
                dev_losses = []
                dev_accs = []

                for j_batch, dev_batched in enumerate(dev_dat_loader):

                    dists = dev_batched['dists']
                    labels = dev_batched['labels']

                    if (config['use_gpu']):
                        dists, labels = dists.cuda(), labels.cuda()

                    model.eval()
                    outputs = model(dists)
            
                    dev_loss = criterion(outputs, labels).cpu().data
                    dev_losses.append(dev_loss)

                    dev_acc = accuracy_score(np.round(outputs.cpu().detach()), labels.cpu())
                    dev_accs.append(dev_acc)

                    logging.info(' Epoch: [%d/%d], Dev Batch %d, Dev Loss: %.4f, Dev Accuracy: %.2f' % (epoch+1, config['num_epochs'], j_batch + 1, dev_loss, dev_acc))

            model.train()

    save_model(epoch, model, optimizer, loss, output_dir)

    return model
