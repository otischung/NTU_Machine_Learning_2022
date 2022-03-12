from import_package import *

# Configurations
# config contains hyper-parameters for training and the path to save your model.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,  # Your seed number, you can pick your lucky number. :)
    'select_all': True,  # Whether to use all features.
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'n_epochs': 5000000,  # Number of epochs.
    'batch_size': 512,
    'learning_rate': 1e-5,
    'early_stop': 50000,  # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}
