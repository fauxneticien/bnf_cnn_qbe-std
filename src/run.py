from helpers import *

config_file = sys.argv[1]

# config_file = 'data/sws2013-sample/test_config.yaml'

# Set up experiment from config file
config     = load_parameters(config_file)
output_dir = setup_exp(config)

# Load datasets
datasets    = load_std_datasets(config['datasets'])
dataloaders = create_data_loaders(datasets, config)

# If configured to load a previous model...
if('model_path' in config.keys()):
    model, optimizer, criterion = load_saved_model(config)
# If no previous model specified then load a new one according to config file
else:
    model, optimizer, criterion = instantiate_model(config)

if(config['mode'] == 'eval'):

    for ds_name, ds_loader in dataloaders.items():

        csv_path = make_results_csv(os.path.join(output_dir, ds_name + '-results.csv'), headers = 'eval')
        logging.info(" Creating output file at '%s'" % (csv_path))

        logging.info(" Starting evaluation on dataset '%s'" % (ds_name))
        
        with torch.no_grad():

            run_model(
                model = model,
                mode = 'eval',
                ds_loader = ds_loader,
                use_gpu = config['use_gpu'],
                csv_path = csv_path,
                keep_loss = False,
                criterion = None,
                epoch = None,
                optimizer = None
            )

elif(config['mode'] == 'train'):

    train_csv_path = make_results_csv(os.path.join(output_dir, 'train_results.csv'), headers = 'train')

    if('dev' in datasets.keys() and 'eval_dev_epoch' in config.keys()):
        dev_csv_path = make_results_csv(os.path.join(output_dir, 'dev_results.csv'), headers = 'train')
    else:
        # If no dev set defined, then no need to evaluate on dev set every nth epoch
        config['eval_dev_epoch'] = None

    for epoch in range(1, config['num_epochs'] + 1):

        model, optimizer, criterion, loss, mean_loss = run_model(
            model = model,
            mode = 'train',
            ds_loader = dataloaders['train'],
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
