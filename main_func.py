import os
import argparse
from solver_m import Solver
from data_loader import get_loader
from torch.backends import cudnn


class ARG:
    def __init__(self,test_iters=200000):
        self.mode = 'test'
        self.dataset = 'RaFD'
        self.image_size = 128   #输出的大小
        self.c_dim = 8  #标签的维度
        self.rafd_image_dir = 'data/RaFD/test'
        self.sample_dir = 'stargan_rafd/samples'
        self.log_dir = 'stargan_rafd/logs'
        self.model_save_dir = 'stargan_rafd/models'
        self.result_dir = 'stargan_rafd/results'
        
        # Model configuration.
        self.c2_dim = 8  #dimension of domain labels (2nd dataset)     
        self.celeba_crop_size = 178  #crop size for the CelebA dataset   
        self.rafd_crop_size = 256 #crop size for the RaFD dataset 数据集的裁剪大小
        self.g_conv_dim = 64  #number of conv filters in the first layer of G
        self.d_conv_dim = 64  #number of conv filters in the first layer of D
        self.g_repeat_num = 6  #number of residual blocks in G
        self.d_repeat_num = 6  #number of strided conv layers in D
        self.lambda_cls = 1
        self.lambda_rec = 10 #weight for reconstruction loss
        self.lambda_gp = 10 #weight for gradient penalty
    
        # Training configuration.
        self.batch_size = 16     #help='mini-batch size
        self.num_iters = 200000    #help='number of total iterations for training D
        self.num_iters_decay = 100000   #, help='number of iterations for decaying lr
        self.g_lr = 0.0001  #help='learning rate for G
        self.d_lr = 0.0001  #help='learning rate for D
        self.n_critic = 5   #help='number of D updates per each G update
        self.beta1 = 0.5   #help='beta1 for Adam optimizer
        self.beta2 = 0.999   #help='beta2 for Adam optimizer
        self.resume_iters = None  #help='resume training from this step
        self.selected_attrs = ['angry', 'contemptuous', 'disgusted', 'fearful', 'happy','neutral','sad','surprised']

        # Test configuration. 
        self.test_iters = test_iters   #help='test model from this step

        # Miscellaneous.
        self.num_workers = 1
        self.use_tensorboard = True

        # Directories.
        self.celeba_image_dir = 'data/celeba/images'
        self.attr_path = 'data/celeba/list_attr_celeba.txt'
        # Step size.
        self.log_step = 10
        self.sample_step = 1000
        self.model_save_step = 10000
        self.lr_update_step = 1000
        

def str2bool(v):
    return v.lower() in ('true')

def main(test_iters):
    # For fast training.
    cudnn.benchmark = True
    
    config = ARG(test_iters)

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = None
    rafd_loader = None
    rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                 config.rafd_crop_size, config.image_size, config.batch_size,
                                 'RaFD', config.mode, config.num_workers)
    

    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, rafd_loader, config)

    if config.mode == 'test':
        if config.dataset in ['RaFD']:
            return solver.test()

    


