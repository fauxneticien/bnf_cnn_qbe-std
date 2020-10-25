import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import accuracy_score

# from class files in src folder
from Datasets import STD_Dataset
from Models import ConvNet

# Set up data
sws2013_sample_train = STD_Dataset(
    root_dir   = 'data/sws2013-sample',
    labels_csv = 'train_labels.csv',
    query_dir  = 'train_queries',
    audio_dir  = 'references'
)

sws2013_sample_test = STD_Dataset(
    root_dir   = 'data/sws2013-sample',
    labels_csv = 'test_labels.csv',
    query_dir  = 'test_queries',
    audio_dir  = 'references'
)

# Set up hyperparameters
num_epochs = 1
learning_rate = 0.001
use_gpu = False

# Set up model
model = ConvNet()

if (use_gpu):
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCELoss()

print('Training model:')

for epoch in range(num_epochs):

    dataloader = DataLoader(
        sws2013_sample_train,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    for i_batch, sample_batched in enumerate(dataloader):
        dists = sample_batched['dists']
        labels = sample_batched['labels']

        if (use_gpu):
            dists, labels = dists.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(dists)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        accuracy = accuracy_score(np.round(outputs.cpu().detach()), labels.cpu())

        if(i_batch % 100 == 0):
            print('Epoch: [%d/%d], Batch: %d, Loss: %.4f, Accuracy: %.2f' % (epoch+1, num_epochs, i_batch, loss.data, accuracy))

print('Testing model:')

dataloader = DataLoader(sws2013_sample_test, batch_size = len(sws2013_sample_test))
model.eval()

for i_batch, sample_batched in enumerate(dataloader):
    outputs = model(sample_batched['dists'])
    accuracy = accuracy_score(np.round(outputs.detach()), sample_batched['labels'])

    print('Test accuracy: %.2f' % accuracy)
