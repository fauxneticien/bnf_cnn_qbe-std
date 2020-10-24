# Query by Example Spoken Term Detection using bottleneck features and a convolutional neural network

This project is a modification of the CNN-based QbE-STD method proposed by Ram, Miculicich, and Bourlard (2018) (see original repo: [https://github.com/idiap/CNN_QbE_STD](https://github.com/idiap/CNN_QbE_STD)).
In this implementation we use bottleneck features instead of phone posteriors used in the original design. 
Bottleneck features were extracted using the Shennong speech features library ([https://github.com/bootphon/shennong](https://github.com/bootphon/shennong)).

## Sample data

A sample of extracted features are included in this repo as numpy `.npy` files (see `data/sws2013-sample`).
Each file is a numpy array of shape (N, 80), where N is the number of frames returned by the feature extractor and 80 the number of activation values. 
The M x N distance matrix between a query of size (M, 80) and reference of size (N, 80) is created on the fly (see `STD_Dataset` class in  `src/Datasets.py`)

The sws2013 database (20+ hours of audio) is too big to include here.
If you want to run the full training process, you can get the original data from [https://speech.fit.vutbr.cz/software/sws-2013-multilingual-database-query-by-example-keyword-spotting](https://speech.fit.vutbr.cz/software/sws-2013-multilingual-database-query-by-example-keyword-spotting).

## Run on sample data

`python src/train_sws2013-sample.py`
