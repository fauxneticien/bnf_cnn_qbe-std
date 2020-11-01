from helpers import *
import re

config_file = sys.argv[1]

# config_file = 'data/sws2013-sample/test_config.yaml'

# Set up experiment from config file
config     = load_parameters(config_file)
output_dir = setup_exp(config)

# Load datasets
datasets    = load_std_datasets(config['datasets'], config['apply_vad'])
dataloaders = create_data_loaders(datasets, config)

if(config['mode'] == 'eval'):

    for ds_name, ds_loader in dataloaders.items():

        logging.info(" Starting evaluation on dataset '%s'" % (ds_name))
        csv_path = make_results_csv(os.path.join(output_dir, ds_name + '-results.csv'))
        logging.info(" Creating output file at '%s'" % (csv_path))

        model_paths = []

        # If configured to load a single model...
        if os.path.isfile(config['model_path']):
            model_paths.append(config['model_path'])

        # If given a directory of model checkpoints
        elif os.path.isdir(config['model_path']):
            model_paths = [ os.path.join(config['model_path'], m) for m in os.listdir(config['model_path']) ]
            model_paths.sort()

        for model_path in model_paths:

            epoch = int(re.search('model-e(\d+).pt', model_path).group(1))
            config['model_path'] = model_path
            
            model, _, _ = load_saved_model(config)

            with torch.no_grad():

                run_model(
                    model = model,
                    mode = 'eval',
                    ds_loader = ds_loader,
                    use_gpu = config['use_gpu'],
                    csv_path = csv_path,
                    keep_loss = False,
                    criterion = None,
                    epoch = epoch,
                    optimizer = None
                )

elif(config['mode'] == 'train'):

    # If configured to load a previous model...
    if('model_path' in config.keys()):
        model, optimizer, criterion = load_saved_model(config)
    # If no previous model specified then load a new one according to config file
    else:
        model, optimizer, criterion = instantiate_model(config)

    train_csv_path = make_results_csv(os.path.join(output_dir, 'train_results.csv'), headers = 'train')

    if('dev' in datasets.keys() and 'eval_dev_epoch' in config.keys()):
        dev_csv_path = make_results_csv(os.path.join(output_dir, 'dev_results.csv'), headers = 'train')
    else:
        # If no dev set defined, then no need to evaluate on dev set every nth epoch
        config['eval_dev_epoch'] = None

    for epoch in range(1, config['num_epochs'] + 1):

        if isinstance(config['datasets']['train']['labels_csv'], dict):
            # If separate positive and negative label CSV files supplied, re-sample negatives at each epoch
            epoch_train_ds = load_std_datasets({'train' : config['datasets']['train']}, config['apply_vad'])
            epoch_train_dl = create_data_loaders(epoch_train_ds, config)['train']
        else:
            epoch_train_dl = dataloaders['train']

        model, optimizer, criterion, loss, mean_loss = run_model(
            model = model,
            mode = 'train',
            ds_loader = epoch_train_dl,
            use_gpu = config['use_gpu'],
            csv_path = train_csv_path,
            keep_loss = True,
            criterion = criterion,
            epoch = epoch,
            optimizer = optimizer
        )

        logging.info(' Epoch: [%d/%d], Train Loss: %.4f' % (epoch, config['num_epochs'], mean_loss))

        if(epoch % config['save_epoch'] == 0):
            save_model(epoch, model, optimizer, loss, output_dir)

        if(config['eval_dev_epoch'] is not None and epoch > 1 and epoch % config['eval_dev_epoch'] == 0):
        
            with torch.no_grad():

                model, optimizer, criterion, loss, mean_loss = run_model(
                    model = model,
                    mode = 'eval',
                    ds_loader = dataloaders['dev'],
                    use_gpu = config['use_gpu'],
                    csv_path = dev_csv_path,
                    keep_loss = True,
                    criterion = criterion,
                    epoch = epoch,
                    optimizer = optimizer
                )

            logging.info(' Epoch: [%d/%d], Dev Loss: %.4f' % (epoch, config['num_epochs'], mean_loss))
