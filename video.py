import os
import yaml
import argparse
import os.path as osp
import torch
from models import *
import numpy as np
#from pytorch_lightning import Trainer
from experiment import SceneFlowExp
from utils.utils import visualize_bev

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = argparse.ArgumentParser(description='Generic runner for Scene-Flow models')
parser.add_argument('--config', '-c',
                        dest='filename',
                        metavar='FILE',
                        help='Path to .yaml config file for the experiment',
                        default='configs/test/flowstep3d_self.yaml')
                    
args = parser.parse_args()


data_root='/GPFS/data/zuhongliu/dair_v2x_preprocess/'

scene_idx=['yizhuang10/train/008348_008349.npy',
           'yizhuang10/train/008349_008350.npy',
           'yizhuang10/train/008350_008351.npy',
           'yizhuang10/train/008351_008352.npy',
           'yizhuang10/train/008352_008353.npy',  
           'yizhuang10/train/008353_008354.npy',           
           ]

with open(args.filename, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)
    #pl.utilities.seed.seed_everything(seed=18)
print(config)

model = models_dict[config['model_params']['model_name']](**config['model_params']).cuda()
experiment = SceneFlowExp(model, config['exp_params'])

for i in range(len(scene_idx)):
    filename=osp.join(data_root,scene_idx[i])
    with open(filename, 'rb') as fp:
        data = np.load(fp,allow_pickle=True)
        data=data.item()
        pc1 = data['pc1'].astype('float32')
        pc2 = data['pc2'].astype('float32')
        flow = data['flow'].astype('float32')

        n1=len(pc1)
        n2=len(pc2)
        #print(n1,n2)
        #print(n2)
            
        '''

        if n1 >= config["exp_params"]["data"]["num_points"]:
            sample_idx1=np.random.choice(np.arange(n1),config["exp_params"]["data"]["num_points"],replace=False)

        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, config["exp_params"]["data"]["num_points"] - n1, replace=True)), axis=0)
            #print(sample_idx1)
        pc1_ = pc1[sample_idx1, :]
        flow_ = flow[sample_idx1, :]

        pc1 = pc1_.astype('float32')
        flow = flow_.astype('float32')

        if n2 >= config["exp_params"]["data"]["num_points"]:
            sample_idx2=np.random.choice(np.arange(n2),config["exp_params"]["data"]["num_points"],replace=False)
        else:
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, config["exp_params"]["data"]["num_points"] - n2, replace=True)), axis=0)
            

        pc2_ = pc2[sample_idx2, :]
        pc2 = pc2_.astype('float32')
    
    '''
    pc1=torch.FloatTensor(pc1).unsqueeze(0).cuda()
    pc2=torch.FloatTensor(pc2).unsqueeze(0).cuda()
    flow=torch.FloatTensor(flow).unsqueeze(0).cuda()


    flow_pred=experiment.forward(pc1,pc2,pc1,pc2,experiment.hparams['test_iters'])
    visualize_bev(pc1,pc2,flow_pred[-1],i)

    

