from argparse import ArgumentParser
from dataset.base import dataroot
parser = ArgumentParser()
DATA = 'nerf_synthetic'
DEVICE = 'cuda:0'
'''
    Data
'''
parser.add_argument(
    '--data',
    type=str,
    default=DATA,
    choices=['dtu', 'llff', 'nerf_synthetic'],
    help='data'
)
parser.add_argument(
    '--data_root',
    type=str,
    default=dataroot(DATA),
    choices=['dtu', 'llff', 'nerf_synthetic'],
    help='the path to data'
)

parser.add_argument(
    '--device',
    type=str,
    default=DEVICE,
    help='device'
)
'''
    Network.DepthToPoints
'''
parser.add_argument(
    '--num_voxel',
    type=int, 
    default=256,
    help='the number of voxels to discretize the space, \
          which is used to determine the depth range in depth-aware interpolation.'
)
'''
    Network.ZbufferSearching
'''
parser.add_argument(
    '--kernel',
    type=int, 
    default=7, # 5 by default -> TODO: ablation study
    help='the kernel size of z-buffer searching.'
)
parser.add_argument(
    '--NSP',
    type=int, 
    default=30,
    help='the number of sample point in z-buffer searching.'
)
parser.add_argument(
    '--NPC',
    type=int, 
    default=8,
    help='the number of point cloud in z-buffer searching.'
)


'''
Training
'''
parser.add_argument(
    '--batch_size',
    type=int,
    default=10240, 
    help='The batch size of training.'
)

parser.add_argument(
    '--total_step',
    type=int,
    default=50, 
    help='The step of training.'
)
parser.add_argument(
    '--val_interval',
    type=int,
    default=1, 
    help='The interval of validation.'
)
parser.add_argument(
    '--status',
    type=str,
    default='train_on_pc', 
    choices=['train_on_pc', 'train_on_grid', 'inference'],
    help='The interval of validation.'
)
parser.add_argument(
    '--lr_decay_iters',
    type=int,
    default=1000000,
    help='The total number of iterations to decay the learning rate.'
)
parser.add_argument(
    '--lr_decay_exp',
    type=float,
    default=0.1,
    help='The learning rate decay factor.'
)
parser.add_argument('--lr_decay', action='store_true', default=True)

# neural_points init
parser.add_argument('--init_sigma', type=float,
                   default=0.1,
                   help='initialization sigma')

parser.add_argument('--epochs', type=float,
                   default=20,
                   help='training epochs')

# save ckpt
parser.add_argument('--ckpt', type=str,
                   help='the name of checkpoint')

parser.add_argument(
    '--optimizer_type',
    type=str,
    default="adam", 
    help='The type of optimizer.'
)
parser.add_argument(
    '--lr_cfg',
    type=dict,
    default={
            "lr_init": 1.0e-3,
            "decay_step": 10,
            "decay_rate": 0.9,
        }, 
    help='The learning rate configuration.'
)
parser.add_argument(
    '--lr_type',
    type=str,
    default="exp_decay", 
    help='The type of learning rate adjustment strategy.'
)