import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as mp
from utils.canvas_bev import Canvas_BEV
import os.path as osp
#from data.data_utils import voxelize_occupy


voxel_size = (1, 1, 0.4)
area_extents = np.array([[0., 128.], [-64., 64.], [-2, 3]])

def debugger_mode():
    import sys
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    elif gettrace():
        return True


def get_num_workers(num_workers):
    if debugger_mode():
        return 0
    else:
        return num_workers


def detach_tensor(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x

def voxelize_occupy(pts,flow, voxel_size, extents=None, return_indices=False):
    """
    Voxelize the input point cloud. We only record if a given voxel is occupied or not, which is just binary indicator.

    The input for the voxelization is expected to be a PointCloud
    with N points in 4 dimension (x,y,z,i). Voxel size is the quantization size for the voxel grid.

    voxel_size: I.e. if voxel size is 1 m, the voxel space will be
    divided up within 1m x 1m x 1m space. This space will be 0 if free/occluded and 1 otherwise.
    min_voxel_coord: coordinates of the minimum on each axis for the voxel grid
    max_voxel_coord: coordinates of the maximum on each axis for the voxel grid
    num_divisions: number of grids in each axis
    leaf_layout: the voxel grid of size (numDivisions) that contain -1 for free, 0 for occupied

    :param pts: Point cloud as N x [x, y, z, i]
    :param voxel_size: Quantization size for the grid, vd, vh, vw
    :param extents: Optional, specifies the full extents of the point cloud.
                    Used for creating same sized voxel grids. Shape (3, 2)
    :param return_indices: Whether to return the non-empty voxel indices.
    """
    # Function Constants
    VOXEL_EMPTY = 0
    VOXEL_FILLED = 1

    # Check if points are 3D, otherwise early exit
    if pts.shape[1] < 3 or pts.shape[1] > 4:
        raise ValueError("Points have the wrong shape: {}".format(pts.shape))

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))

        filter_idx = np.where((extents[0, 0] < pts[:, 0]) & (pts[:, 0] < extents[0, 1]) &
                              (extents[1, 0] < pts[:, 1]) & (pts[:, 1] < extents[1, 1]) &
                              (extents[2, 0] < pts[:, 2]) & (pts[:, 2] < extents[2, 1]))[0]
        pts = pts[filter_idx]

    # Discretize voxel coordinates to given quantization size
    discrete_pts = np.floor(pts[:, :3] / voxel_size).astype(np.int32)

    # Use Lex Sort, sort by x, then y, then z
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    z_col = discrete_pts[:, 2]
    sorted_order = np.lexsort((z_col, y_col, x_col))

    # Save original points in sorted order
    discrete_pts = discrete_pts[sorted_order]

    # Format the array to c-contiguous array for unique function
    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1])))

    # The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # Sort unique indices to preserve order
    unique_indices.sort()

    voxel_coords = discrete_pts[unique_indices]

    if flow is not None:
        #print("flow:",flow)
        #print("unique_indices:",unique_indices.shape)
        num_points_in_voxel = np.diff(unique_indices)
        #print("num_points_in_voxel:",num_points_in_voxel.shape)
        
        flow=flow[sorted_order]
        #print("discrete,points:",discrete_pts)
        #print("flow:",flow)
        mask=np.stack([unique_indices[:-1],unique_indices[:-1]+num_points_in_voxel],axis=1)
        #mask=np.append(mask,np.array([[unique_indices[-1],-1]]),axis=0)
        #print("mask:",mask)
        #mask=np.append
        #print(flow[mask[:]].shape)
        #for i in range(mask.shape[0]):
        #    print(flow[mask[i,0]:mask[i,1]])

        disp_list=[np.mean(flow[mask[i,0]:mask[i,1]],axis=0) for i in range(mask.shape[0])]
        disp_list.append(flow[unique_indices[-1]])
        #print(disp_list[0].shape)
        disp=np.stack(disp_list,axis=0)/voxel_size    
        
        #print("disp:",np.sum(disp[:,0]))
        #disp_field=np.zeros_like((unique_indices.shape[0],2))
        #disp_field
        #print(unique_indices[:-1].shape)
        #print(num_points_in_voxel.shape)
        #print("discrete_pts:",discrete_pts.shape)
        #print(unique_indices[-1])
        #print("mask:",mask.shape)
        #disp_field=np.concatenate([np.mean(flow[unique_indices[i]:unique_indices[i]+num_points_in_voxel[i]],axis=0) for i in range(num_points_in_voxel.shape[0])],axis=0)
        #print(disp.shape)
        #funique_indices[:-1]
        #unique_indices[:-1]+num_points_in_voxel[:]
        #print(flow_field.shape)
    else:
        disp=None

    # Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_voxel_coord = np.floor(extents.T[0] / voxel_size)
        max_voxel_coord = np.ceil(extents.T[1] / voxel_size) - 1
    else:
        min_voxel_coord = np.amin(voxel_coords, axis=0)
        max_voxel_coord = np.amax(voxel_coords, axis=0)

    # Get the voxel grid dimensions
    num_divisions = ((max_voxel_coord - min_voxel_coord) + 1).astype(np.int32)

    # Bring the min voxel to the origin
    voxel_indices = (voxel_coords - min_voxel_coord).astype(int)

    # Create Voxel Object with -1 as empty/occluded
    leaf_layout = VOXEL_EMPTY * np.ones(num_divisions.astype(int), dtype=np.float32)

    # Fill out the leaf layout
    leaf_layout[voxel_indices[:, 0],
                voxel_indices[:, 1],
                voxel_indices[:, 2]] = VOXEL_FILLED

    if return_indices:
        return leaf_layout, voxel_indices,disp
    else:
        return leaf_layout,disp

def visualize(pc1,pc2,flow_pred):
    pc_range = [0, -50, -10, 100, 50, 10]
    canvas = Canvas_BEV(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                                canvas_x_range=(pc_range[0], pc_range[3]), 
                                                canvas_y_range=(pc_range[1], pc_range[4]),
                                                left_hand=False
                                            )
    if type(pc1)==torch.Tensor:    
       
        pc1=pc1.cpu().numpy()[0]
        pc2=pc2.cpu().numpy()[0]
        flow_pred=flow_pred.detach().cpu().numpy()[0]
    
    pc1_deformed=pc1+flow_pred
    #canvas_xy, valid_mask = canvas.get_canvas_coords(pc1) # Get Canvas Coords
    # print(canvas_xy.shape)
    # print(valid_mask.shape)
    #color='Blues_r' 
    #canvas.draw_canvas_points(canvas_xy[valid_mask],colors=color) # Only draw valid points
    canvas_xy, valid_mask = canvas.get_canvas_coords(pc2) # Get Canvas Coords
    # print(canvas_xy.shape)
    # print(valid_mask.shape)
    color="OrRd_r"
    canvas.draw_canvas_points(canvas_xy[valid_mask],colors=color) # Only draw valid points
    canvas_xy, valid_mask = canvas.get_canvas_coords(pc1_deformed) # Get Canvas Coords
    # print(canvas_xy.shape)
    # print(valid_mask.shape)
    color="Greens"
    canvas.draw_canvas_points(canvas_xy[valid_mask],colors=color) # Only draw valid points

    plt.axis("off")

    plt.imshow(canvas.canvas)

    plt.tight_layout()
    save_path = osp.join("/GPFS/rhome/zuhongliu/SSL_OCC/SSL_Flow/flowstep3d/vis", "pc.png")
    plt.savefig(save_path, transparent=False, dpi=400)
    plt.clf()


def visualize_bev(pc1,pc2,flow_pred,i):
    if type(pc1)==torch.Tensor:
        pc1=pc1.cpu().numpy()[0]
        pc2=pc2.cpu().numpy()[0]
        flow_pred=flow_pred.detach().cpu().numpy()[0]

    r1,indice1,_=voxelize_occupy(pc1,None,voxel_size=voxel_size,extents=area_extents,return_indices=True)
    r2,indice2,_=voxelize_occupy(pc2,None,voxel_size=voxel_size,extents=area_extents,return_indices=True)

    past_heatmap=np.mean(r1,axis=-1)

    
    idx_x = np.arange(past_heatmap.shape[0])
    idx_y = np.arange(past_heatmap.shape[1])
    idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
    img=mp.imshow(past_heatmap,cmap='jet')
    fig=plt.gcf()
    #p="/"+str(cnt)
    #os.mkdir(p)
    fig.savefig("./vis/video/bev_"+str(i)+".jpg")

    heatmap=np.mean(r2,axis=-1)
        
    img=mp.imshow(heatmap,cmap='jet')
    fig=plt.gcf()
    #p="/"+str(cnt)
    #os.mkdir(p)
    fig.savefig("./vis/video/bev_"+str(i+1)+".jpg")
        
    pc1_deformed=pc1+flow_pred
    #pc1_deformed=torch.cat([motion_pc1_deformed,static_pc1],dim=1).cpu().numpy()
        
    res_pred,_,_=voxelize_occupy(pc1_deformed,None,extents=area_extents,voxel_size=voxel_size,return_indices=True)

    heatmap_pred=np.mean(res_pred,axis=-1)
        
    img=mp.imshow(heatmap_pred,cmap='jet')
    fig=plt.gcf()
    #p="/"+str(cnt)
    #os.mkdir(p)
    fig.savefig("./vis/video/bev_"+str(i+1)+"_deformed.jpg")