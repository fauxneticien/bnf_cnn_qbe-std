# Query by example spoken term detection using bottleneck features and a convolutional neural network

This project is a modification of the CNN-based QbE-STD method proposed in [Ram, Miculicich, and Bourlard (2018)](https://pdfs.semanticscholar.org/5533/7b94a7d2f3a16c71d7852cff290a45637e22.pdf), with code available on [https://github.com/idiap/CNN_QbE_STD](https://github.com/idiap/CNN_QbE_STD).
In this implementation we use bottleneck features instead of phone posteriors used in the original design. 

Bottleneck features were extracted using the [Shennong speech features library](https://github.com/bootphon/shennong), which provides a Python API for the [BUT/Phonexia bottleneck features extractor](https://speech.fit.vutbr.cz/software/but-phonexia-bottleneck-feature-extractor) trained on 17 languages from the IARPA Babel dataset (Cantonese, Pashto, Turkish, Tagalog, Vietnamese, Assamese, Bengali, Haitian Creole, Lao, Tamil, Zulu, Kurdish, Tok Pisin, Cebuano, Kazach, Telugu, Lithuanian).

## Usage

To run training or inference, supply a config file with the required information.
There are two sample configuration files included in `data/sws2013-sample` (for more info see below).

```python
python src/run.py <config_file>
```

**Requirements**: Python 3.7, PyTorch 1.6.0, etc.* (*experiments were run on a [Genesis Cloud](https://www.genesiscloud.com/) instances with Python, PyTorch and CUDA pre-configured on launch). 

## Sample files

A set of sample files from the Spoken Web Search 2013 (SWS2013) corpus is included here in the `data/sws2013-sample` directory.
The sws2013 database (20+ hours of audio) is too big to include here.
If you want to run the full training process, you can get the original data from the [BUT Speech@Fit website](https://speech.fit.vutbr.cz/software/sws-2013-multilingual-database-query-by-example-keyword-spotting) and use the feature extractor provided in `src/extract_bnf.py`.

```
data/sws2013-sample
├── train_queries     <- Queries for training
├── test_queries      <- Queries for development/testing (same for sample data)
├── Audio             <- Corpus in which dev/test queries are searched for
├── train_labels.csv  <- Ground truth for train_queries
|                        (1 = query occurs in reference, 0 = otherwise)
├── dev_labels.csv    <- Ground truth for dev/test queries
├── train_config.yaml <- Config file to run training on sws2013-sample data
├── test_config.yaml  <- Config file to run inference on sws2013-sample data 
|                        using model saved at the 60th epoch from training process
```

Running `python src/run.py data/sws2013-sample/train_config.yaml` followed by `python src/run.py data/sws2013-sample/test_config.yaml` should result in the output process shown below:

![Sample process](data/sws2013-sample/sample_run.gif)

## Evaluation

The primary outputs from `run.py` generate CSV files shown below, where query, reference, and label are those given in the ground truth label files (e.g. `train_labels.csv`) and the prediction column the output of the CNN.
For CSV files generated from the training process, there is an additional epoch column.

| Query | Reference | Label | Prediction |
| ------------- | ------------- | ------------- | ------------- |
| sws2013_dev_221_07  | sws2013_04169 | 0 | 0.68009675 |
| sws2013_dev_391_06  | sws2013_03545 | 1 | 0.83309245 |
| ... | ... | ... | ...

Following the original SWS2013 evaluation metrics and those also reported in [Ram, Miculicich, and Bourlard (2018)](https://pdfs.semanticscholar.org/5533/7b94a7d2f3a16c71d7852cff290a45637e22.pdf), the Maximum Term Weighted Value (MTWV) is used as the primary evaluation metric.
To retrieve the Actual and Maximum Term Weighted Values, use the script provided in `src/mtwv.R`.
The script returns CSV data (to stdout) listing the ATWVs at different thresholds sorted by largest ATWVs (i.e. the MTWV) to smallest.
Also included are average precision and average recall as secondary metrics.

| Threshold | ATWV | Average Precision | Average Recall |
| ------------- | ------------- | ------------- | ------------- |
| 0.70  | **0.683** | 0.0419 | 0.689 |
| 0.75  | 0.652 | 0.0449 | 0.657 |
| ... | ... | ... | ...
