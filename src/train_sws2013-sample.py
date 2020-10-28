# from class and helper files in src folder
from Datasets import STD_Dataset
from Models import *
from helpers import *

# train_config = load_parameters('data/sws2013-sample/train_config.yaml')

train_config = load_parameters('data/sws2013/train_config.yaml')

model  = train_model(train_config)
