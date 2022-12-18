import os
import numpy as np
from utils.utils import visualize

filename='/GPFS/data/zuhongliu/dair_v2x_preprocess/yizhuang08/val/000497_000498.npy'

with open(filename, 'rb') as fp:
    data = np.load(fp,allow_pickle=True)
    data=data.item()
    pc1 = data['pc1'].astype('float32')
    pc2 = data['pc2'].astype('float32')
    flow = data['flow'].astype('float32')

print(len(pc1))
print(len(pc2))
visualize(pc1,pc2,flow)

