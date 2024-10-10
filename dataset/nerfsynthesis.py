import os, sys;sys.path.append(os.getcwd())

import torch
import json
import time
import numpy as np
from imageio import imread
from torch.utils.data import Dataset
from utils.utils import color_map_backward, resize_img, downsample2ratio, totensor


class NeRFSynthesis(Dataset):
    '''
    Neural Radiance Fields for View Synthesis".

    Stats:
    + 8 Scenes
    + 100 Training images
    + 100 Validation images
    + 200 Test images
    + Images are 800x800

    Structure:
    SCENE_NAME
        -train
        r_*.png
        -val
        r_*.png
        -test
        r_*.png
        r_*_depth_0000.png
        r_*_normal_0000.png
        transforms_train.json
        transforms_val.json
        transforms_test.json
    '''
    def __init__(self,
                 args,
                 scene,
                 ref_downsample=1,
                 src_downsample=2,
                 background='white',
                 split='train',
                 num_src=4,
                 status = 'train_on_pc',
                 pc_type=None
                 ):
        assert status in ['train_on_pc', 'train_on_grid', 'inference']
        self.status = status
        self.root_dir = os.path.join(args.data_root, scene)
        self.scene = scene
        self.background = background
        self.ref_downsample = ref_downsample
        self.src_downsample = src_downsample
        self.num_src = num_src
        self.pc_type = pc_type
        if isinstance(split, list):
            self.img_ids, self.w2cs, self.c2ws, self.K = [], [], [], []
            for spl in split:
                img_ids, w2cs, c2ws, K = self.parse_info(split=spl, num_src=num_src)
                self.img_ids.extend(img_ids)
                self.w2cs.extend(w2cs)
                self.c2ws.extend(c2ws)
                self.K = K
        else:
            self.img_ids, self.w2cs, self.c2ws, self.K = self.parse_info(split=split, num_src=num_src)
                
        self.ori_hw = np.array((800, 800))
        self.znear, self.zfar = 2, 6
        if self.status == 'train_on_pc':
            self.split_ray()
        if pc_type == 'pointnerf':
            self.pc_path = os.path.join('pointnerf_ckpt//nerf_synthesis', self.scene+'.pth')
        else:
            self.pc_path = f'/home/jiahao/nerf/data/nerfs/nerf_sythetic/{scene}/point_clouds/pointclouds.ply'
    def split_ray(self, grid_h=64, grid_w=64, shuffle=True):
        self.ray_id_list = []
        ray_index_x, ray_index_y = torch.meshgrid( torch.arange(0, self.ori_hw[1], dtype=torch.int64), torch.arange(0, self.ori_hw[0], dtype=torch.int64), indexing='xy')
        ray_index = torch.stack((ray_index_x, ray_index_y), dim=-1).contiguous().view(-1, 2)
        if isinstance(self.img_ids, list):
            num_img = len(self.img_ids)
        elif isinstance(self.img_ids, np.ndarray):
            num_img = self.img_ids.shape[0]
        else:
            raise ValueError('img_ids should be list or np.ndarray')
        for img_idx in range(num_img):

            ray_index_ = ray_index[torch.randperm(ray_index.shape[0])] if shuffle else ray_index
            ray_index_ = torch.cat([ray_index_, torch.ones((ray_index_.shape[0], 1), dtype=torch.int64) * img_idx], dim=-1)
            
            ray_index_ = ray_index_.split(split_size=grid_h*grid_w, dim=0)
            self.ray_id_list.extend([*ray_index_])


    def parse_info(self, split='train', num_src=4): 
        with open(os.path.join(self.root_dir, f'transforms_{split}.json'),'r') as f:
            img_info=json.load(f)
            focal=float(img_info['camera_angle_x'])
            img_ids,w2cs,c2ws = [],[],[]
            temp = '/'.join(img_info['frames'][0]['file_path'].split('/')[1:])
            grid_h,grid_w,_=imread(f'{self.root_dir}/{temp}.png').shape
            focal = .5 * grid_w / np.tan(.5 * focal)
            K=np.asarray([[focal,0,grid_w/2],[0,focal,grid_h/2],[0,0,1]],np.float32)
            for frame in img_info['frames']:
                img_ids.append('-'.join(frame['file_path'].split('/')[1:])) 
                c2w= np.array(frame['transform_matrix'], dtype=np.float32) @ np.diag(np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)) # blender to opencv
                w2c= np.linalg.inv(c2w)
                w2cs.append(w2c[:3, :4])
                c2ws.append(c2w)
        c2ws = np.stack(c2ws, axis=0)
        w2cs = np.stack(w2cs, axis=0)
        # view_id_list = search_nearest_poses(c2ws, num_src)
        
        return np.array(img_ids), totensor(w2cs), totensor(c2ws), totensor(K)#, view_id_list
    
    def img_id2img_path(self, img_id):
        return '//'.join(img_id.split('-'))
    
    def get_image(self, img_id, ratio, normalize=True):
        img = imread(f'{self.root_dir}/{self.img_id2img_path(img_id)}.png')
        alpha = img[:,:,3:].astype(np.float32)/255.0
        img = img[:,:,:3]
        if self.background=='black':
            img = img.astype(np.float32)/255.0
            img = img * alpha
            img = color_map_backward(img)
        elif self.background=='white':
            img = img.astype(np.float32)/255.0
            img = img*alpha + 1.0-alpha
            img = color_map_backward(img)
        else:
            raise NotImplementedError
        
        if ratio != 1.0:
            img = resize_img(img, ratio)
        
        if normalize:
            img = img.astype(np.float32) / 255
            
        return img, alpha
    
    def __len__(self):
        if self.status != 'train_on_pc':
            return len(self.img_ids)
        else:
            return len(self.ray_id_list)
    
    def __getitem__(self, index):
        if self.status == 'train_on_pc':
            ray_id = self.ray_id_list[index]
            ray_index, ref_id = ray_id[:, 0:2],  ray_id[0, 2] # the ref_id is the same for all rays in the same batch so select the first one
        else:
            ray_index = torch.tensor([]) # empty tensor
            ref_id = index
        ref_img_id = self.img_ids[ref_id]
        ref_ratio = downsample2ratio(self.ref_downsample)
        ref_image, ref_mask= self.get_image(ref_img_id, ratio=ref_ratio)
        ref_image = torch.from_numpy(ref_image)
        ref_mask = torch.from_numpy(ref_mask)
        # ref_depth = None
        ref_w2c = self.w2cs[ref_id]
        ref_c2w = self.c2ws[ref_id]
        ref_K = torch.diag(torch.tensor([ref_ratio, ref_ratio, 1])) @ self.K 
        ref_hw = torch.tensor(ref_image.shape[:2])
        if self.status == 'train_on_pc':
            # debug 
            # ray_index = torch.tensor([[400, 400]]).reshape(1, 2)
            ref_rgb = ref_image[ray_index[:, 1], ray_index[:, 0]]
        else:
            ref_rgb = ref_image
        sample = dict(ref_rgb=ref_rgb,
                    ref_w2c=ref_w2c, 
                    ref_c2w=ref_c2w, 
                    ref_K=ref_K,
                    ref_view_id=ref_id,
                    ref_hw=ref_hw,
                    valid_mask = ref_mask,
                    near=torch.tensor([self.znear]),
                    far=torch.tensor([self.zfar]),
                    pc_path=self.pc_path,
                    bg=torch.tensor([1., 1., 1.]) if self.background == 'white' else torch.tensor([0., 0., 0.]),
                    training_ray_index = ray_index,
                    ref_id = ref_id
                    )
      
        return sample
if __name__ == '__main__':
    from utils.config import parser
    args = parser.parse_args()
    nerf_synthesis = NeRFSynthesis(args, scene='lego', split=['train', 'val', 'test'])
    nerf_synthesis[0]