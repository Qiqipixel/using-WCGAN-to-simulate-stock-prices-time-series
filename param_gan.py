# -*- coding: utf-8 -*-
"""
GAN生成模型GAN参数

"""

#%% Parameters

# Data Preprocessing Parameters
use_ret = True # use return or price
scale_ret = 100 # the scale size of return：gan-100

# Network training parameters
gan_type = 'gan' # gan
device = 'cuda' # cuda or cpu
num_epochs = 1000
batch_size = 24 # Takahashi(2019)为24，Koshiyama(2019)为252
seed = 42 # 42
batch_norm = False # gan-False更好
label_size = 40 # the length of label_size(the length of the conditional return series)
critic_iterations=5 #wgan parameters
lamda=10 #wgan parameters
# simulation parameter
num_gen = 100 # the number of the generative return series

# Network structure parameters
if gan_type in ['gan']:
    # GAN structure parameters based on Takahashi(2019)
    gan_dim_latent = 100
    gan_dim_hidden_1 = 256
    gan_dim_hidden_2 = 256 # 2048
    gan_dim_output = 20 # the length of the generative return series
    

# %%
