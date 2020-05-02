## MNIST
############### Configuration file for Bayesian ###############
# n_epochs = 50 ## MNIST
n_epochs = 200 ## FashionMNIST

# lr_start = 0.005
lr_start=0.001
# Check layers stdev for cIFAr and MNISt experiments
# Change the following based on the dataset
# self.log_alpha.data.fill_(-5.0)
# self.log_alpha.data.fill_(0.5)

num_workers = 4
valid_size = 0.2
batch_size = 256 ## FashionMNIST
train_ens = 1
valid_ens = 1

record_mean_var = False
recording_freq_per_epoch = 32
record_layers = ['fc3']

# Cross-module global variables
mean_var_dir = None
record_now = False
curr_epoch_no = None
curr_batch_no = None

##########

# # Peter config
# n_epochs = 30
# lr_start = 0.001