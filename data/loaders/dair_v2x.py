import sys, os
import os.path as osp
import numpy as np

import torch.utils.data as data

__all__ = ['Dair_v2x']


class Dair_v2x(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """

    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 scene=None):
        #assert train is False
        self.train = train
        self.root = data_root
        self.transform = transform
        self.num_points = num_points
        #self.remove_ground = remove_ground
        self.remove_ground=False

        if scene=='all':
            paths=['yizhuang02','yizhuang06','yizhuang08','yizhuang09','yizhuang10','yizhuang13','yizhuang16']
        else:
            paths=[scene]
        self.samples=[]
        for path in paths:
            if self.train:
                tmp_path=osp.join(path,'train')
            else:
                tmp_path=osp.join(path,'val')
            #print("root:",self.root)
            #print("tmp_path:",tmp_path)
            for d in os.listdir(osp.join(self.root,tmp_path)):
                self.samples.append(osp.join(tmp_path,d))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_loaded, pc2_loaded,flow = self.pc_loader(self.samples[index])
        #pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        pc1_transformed, pc2_transformed, sf_transformed = pc1_loaded, pc2_loaded,flow
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        return pc1_transformed, pc2_transformed, pc1_norm, pc2_norm, sf_transformed, self.samples[index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str

    def pc_loader(self, path):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        filename=os.path.join(self.root,path)

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
            

            if n1 >= self.num_points:
                #sample_idx1_flow = np.random.choice(mask1_flow, num_points1_flow, replace=False)
                #try:  # ANCHOR: nuscenes has some cases without nonrigid flows.
                #    sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=False)
                #except:
                #    sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=True)
                #sample_idx1 = np.hstack((sample_idx1_flow, sample_idx1_noflow))
                sample_idx1=np.random.choice(np.arange(n1),self.num_points,replace=False)

            else:
                sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.num_points - n1, replace=True)), axis=0)
           #print(sample_idx1)
            
            pc1_ = pc1[sample_idx1, :]
            flow_ = flow[sample_idx1, :]

            pc1 = pc1_.astype('float32')
            flow = flow_.astype('float32')

            if n2 >= self.num_points:
                #if int(num_points * nonrigid_rate) > len(mask2_flow):
                #    num_points2_flow = len(mask2_flow)
                #    num_points2_noflow = num_points - num_points2_flow
                #else:
                #    num_points2_flow = int(num_points * nonrigid_rate)
                #    num_points2_noflow = int(num_points * rigid_rate) + 1
                #sample_idx2_flow = np.random.choice(mask2_flow, num_points2_flow, replace=False)
                #sample_idx2_noflow = np.random.choice(mask2_noflow, num_points2_noflow, replace=False)
                #sample_idx2 = np.hstack((sample_idx2_flow, sample_idx2_noflow))
                sample_idx2=np.random.choice(np.arange(n2),self.num_points,replace=False)
            else:
                sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.num_points - n2, replace=True)), axis=0)
            

            pc2_ = pc2[sample_idx2, :]
            pc2 = pc2_.astype('float32')
        
        
        return pc1, pc2,flow