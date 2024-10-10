import torch
import os, sys; sys.path.append(os.getcwd())
import imageio
import random
import numpy as np
from copy import copy
from glob import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from kornia.geometry.epipolar import KRt_from_projection
import matplotlib.pyplot as plt
import torch.nn.functional as F


'''
    Utils
'''
def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data

def tensor2list(tensor, datatype=np.int32):
    return tensor.cpu().numpy().astype(datatype).tolist()
    

class DTU(Dataset):
    def __init__(self, 
                 args,
                 path, 
                 scene, 
                 stage='val',
                 ref_downsample=4,
                 src_downsample=4,
                 num_src=4,
                 val_ids=[ 7, 12, 17, 22, 27, 32, 37, 42, 47 ],
                 depth_type='dense',
                 num_images=64,
                 ):
        super().__init__()
        assert ref_downsample >= 1, 'downsample should be larger than 1'
        assert src_downsample >= 1, 'downsample should be larger than 1'
        self.path = path 
        self.scene = scene if isinstance(scene, list) else [scene]
        if len(self.scene) == 1:
            self.pc_path = self.load_point_cloud(os.path.join(self.path, self.scene[0])) 
        else:
            self.pc_path = {scene: self.load_point_cloud(os.path.join(self.path, scene)) for scene in self.scene}
        self.num_src = num_src
        
        self.ref_scale_factor = 1.0 / ref_downsample
        self.src_scale_factor = 1.0 / src_downsample
        self.ref_scale_matrix = torch.tensor([[self.ref_scale_factor, 0, 0],
                                              [0, self.ref_scale_factor, 0],
                                              [0, 0, 1]])
        
        self.src_scale_matrix = torch.tensor([[self.src_scale_factor, 0, 0],
                                              [0, self.src_scale_factor, 0],
                                              [0, 0, 1]])
        
        pair_file = os.path.join(self.path, 'pair.txt')
        pair_data = read_pair_file(pair_file)
        self.depth_type = depth_type
        self.fixed_src_cam_ids = [19, 33, 48, 49]#[19, 33, 48, 49] # [50] # [19, 50] #[19, 33, 50] [19, 33, 48, 49] [19, 33, 48, 49, 1] [19, 33, 48, 49, 50, 1]
        print("selected src cam ids: ", self.fixed_src_cam_ids)

        if stage == 'val':
            self.cam_ids = val_ids
        elif stage == 'train':
            self.cam_ids = list(range(49))
            for ids in val_ids:
                if ids in self.cam_ids:
                    self.cam_ids.remove(ids)

        
        self.znear = torch.tensor(425.0)
        self.zfar = torch.tensor(1000.0)
        self.read_metas(pair_data, num_images)
        # generate point cloud to determine the range of pointcloud for each scene
    
    def cal_range(self, scene, src_cam_ids, src_Ks, src_c2ws):
        src_depth_dir = [os.path.join(self.path, scene, 'depths', f'{cam_id:03d}_{self.depth_type}.npy') for cam_id in src_cam_ids]
        src_depths = torch.stack([self.read_depth(p) for p in src_depth_dir], dim=0)
        _, vxl_l = self.depth2point(src_depths, src_Ks, src_c2ws)
        return vxl_l

    def read_metas(self, pair_data, num_images=64):
        '''
            read the path of depth, image, mask and camera parameters, only read 49 images to train
        '''
        self.metas = list()
        for scene in self.scene:
            for cam_id in self.cam_ids:
                if self.fixed_src_cam_ids is None:
                    self.metas.append(dict(scene = scene, ref_cam_id = cam_id, src_cam_ids = pair_data[cam_id][1][:self.num_src]))
                else:
                    self.metas.append(dict(scene = scene, ref_cam_id = cam_id, src_cam_ids = self.fixed_src_cam_ids))
        
        self.cam_infos = dict()
        default_scale_matrix =  torch.tensor([[0.25, 0, 0],
                                              [0, 0.25, 0],
                                              [0, 0, 1]]) 
        for scene in self.scene:
            _, Ks, w2cs, c2ws, hw = self.load_data(os.path.join(self.path, scene), num_images) # K, w2c, c2w, hw are the same for all scenes
            self.cam_infos[scene] = {'Ks':Ks, 'w2cs':w2cs, 'c2ws':c2ws, 'hw':hw}
            if self.fixed_src_cam_ids is not None:
                src_cam_ids = self.fixed_src_cam_ids
            else:
                src_cam_ids = pair_data[cam_id][1][:self.num_src]
            # the `Ks` is the intrinsic parameters of original size of image (1200, 1600)
            # the depth map we store is (300, 400). So we need to scale the Ks to the size of depth map
            K = default_scale_matrix @ Ks[src_cam_ids]
            self.cam_infos[scene]['vxl_l'] = self.cal_range(scene, src_cam_ids, K, c2ws[src_cam_ids])
        
    def load_data(self, scene_path, num=64):
        img_file = os.path.join(scene_path, 'image', f'{0:06d}.png')
        # num = len(os.listdir(os.path.join(scene_path, 'image')))
        h, w, _ = imageio.imread(img_file).shape
        # h = int(h * self.scale_factor)
        # w = int(w * self.scale_factor)
        
        all_cam = np.load(os.path.join(scene_path, "cameras.npz"))

        w2cs = []
        c2ws = []
        Ks = []
        
        for i in range(num):
            P = all_cam["world_mat_" + str(i)]
            P = P[:3]

            K, R, t = KRt_from_projection(torch.tensor(P)[None])
            K = (K / K[:, 2, 2]).squeeze(0)

            w2c = torch.cat([R.squeeze(0), t.squeeze(0)], dim=-1)
            w2c = torch.cat([w2c, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)])
            c2w = torch.inverse(w2c)
            
            c2ws.append(c2w)
            w2cs.append(w2c)
            Ks.append(K)

        Ks = torch.stack(Ks)#.mean(dim=0)
        w2cs = torch.stack(w2cs)
        c2ws = torch.stack(c2ws)

        return torch.arange(num), Ks, w2cs, c2ws, torch.tensor([h, w])
    
    def load_point_cloud(self, scene_path):
        return os.path.join(scene_path, "mvs_pc.ply")
        
    def load_depth(self, scene_path, depth_type='sparse'):
        assert depth_type in ['dense', 'sparse']
        depth_dir = os.path.join(scene_path, 'depths') 
        depth_dir = sorted(glob(os.path.join(depth_dir, f'*{depth_type}.npy')))
        return depth_dir

    def read_depth(self, depth_dir, scale_factor=0.25):
        if scale_factor == 0.25:
            depth = torch.from_numpy(np.load(depth_dir)[None, :, :])
            return depth
        else:
            scale = scale_factor / 0.25
            depth = torch.from_numpy(np.load(depth_dir)[None, None, :, :])
            depth = F.interpolate(depth, scale_factor=scale, mode='nearest')    
            return depth.squeeze(0)
    
    def load_image(self, scene_path):
        img_dir = os.path.join(scene_path, 'image') 
        img_dir = sorted(glob(os.path.join(img_dir, '*.png')))
        return img_dir
    
    def read_image(self, image_dir, hw):
        image: Image = pil_loader(image_dir)
        image = image.resize((hw[1], hw[0]), Image.BILINEAR)
        image = torch.tensor(np.array(image)).permute(2, 0, 1).contiguous()
        image = image / 255
        return image
    
    def load_mask(self, scene_path):
        mask_dir = os.path.join(scene_path, 'mask') 
        mask_dir = sorted(glob(os.path.join(mask_dir, '*.png')))
        return mask_dir

    def read_mask(self, mask_dir, hw):
        mask: Image = pil_loader(mask_dir)
        mask = mask.resize((hw[1], hw[0]), Image.BILINEAR)
        mask = torch.tensor(np.array(mask), dtype=torch.float32)
        mask = mask.mean(dim=2)[None, ...] / 255
        return mask
    
    def __getitem__(self, index):
        meta = self.metas[index]
        hw = self.cam_infos[meta['scene']]['hw']
        ref_hw = tensor2list(hw * self.ref_scale_factor)
        src_hw = tensor2list(hw * self.src_scale_factor)
        # read reference image, depth, mask and camera parameters
        ref_image_dir = os.path.join(self.path, meta['scene'], 'image', f'{meta["ref_cam_id"]:06d}.png')
        ref_image = self.read_image(ref_image_dir, ref_hw)

        ref_depth_dir = os.path.join(self.path, meta['scene'], 'depths', f'{meta["ref_cam_id"]:03d}_{self.depth_type}.npy')
        ref_depth = self.read_depth(ref_depth_dir, self.ref_scale_factor)

        ref_mask_dir = os.path.join(self.path, meta['scene'], 'mask', f'{meta["ref_cam_id"]:03d}.png')
        ref_mask = self.read_mask(ref_mask_dir, ref_hw)

        ref_K = self.cam_infos[meta['scene']]['Ks'][meta['ref_cam_id']]
        ref_K = torch.matmul(self.ref_scale_matrix, ref_K)
        ref_c2w  = self.cam_infos[meta['scene']]['c2ws'][meta['ref_cam_id']]
        ref_w2c = self.cam_infos[meta['scene']]['w2cs'][meta['ref_cam_id']]

        # read source images, depths, masks and camera parameters
        src_image_dir = [os.path.join(self.path, meta['scene'], 'image', f'{cam_id:06d}.png') for cam_id in meta['src_cam_ids']]
        src_depth_dir = [os.path.join(self.path, meta['scene'], 'depths', f'{cam_id:03d}_{self.depth_type}.npy') for cam_id in meta['src_cam_ids']]
        src_image = torch.stack([self.read_image(p, src_hw) for p in src_image_dir], dim=0)
        src_depth = torch.stack([self.read_depth(p, self.src_scale_factor) for p in src_depth_dir], dim=0)
        src_Ks = self.cam_infos[meta['scene']]['Ks'][meta['src_cam_ids']]
        src_Ks = torch.matmul(self.src_scale_matrix, src_Ks)
        src_c2ws = self.cam_infos[meta['scene']]['c2ws'][meta['src_cam_ids']]
        src_w2cs = self.cam_infos[meta['scene']]['w2cs'][meta['src_cam_ids']]

        
        sample = dict(ref_rgb=ref_image,
                      ref_depth=ref_depth,
                      ref_w2c=ref_w2c, 
                      ref_c2w=ref_c2w, 
                      ref_K=ref_K,
                      ref_view_id=meta["ref_cam_id"],
                      src_view_ids=meta["src_cam_ids"],
                      src_rgbs=src_image,
                      src_depths=src_depth,
                      src_w2cs=src_w2cs,
                      src_c2ws=src_c2ws,
                      src_Ks=src_Ks,
                      ref_hw=ref_hw,
                      src_hw=src_hw,
                      valid_mask = ref_mask,
                      near=torch.tensor([self.znear]),
                      far=torch.tensor([self.zfar]),
                      pc_path=self.pc_path[meta['scene']] if isinstance(self.pc_path, dict) else self.pc_path
                      )
        return sample
    
    def pump_realtime_render_dataset(self):
        ref_rgb = []
        ref_depth = []
        ref_w2c=[] 
        ref_c2w=[]
        ref_K=[]
        ref_view_id=[]
        valid_mask = []
        sample = dict()
        for i, cam_id in enumerate(range(len(self.cam_ids))):
            sample_ = self.__getitem__(cam_id)
            if i == 0:
                sample['src_rgbs'] = sample_['src_rgbs']
                sample['src_depths'] = sample_['src_depths']
                sample['src_w2cs'] = sample_['src_w2cs']
                sample['src_c2ws'] = sample_['src_c2ws']
                sample['src_Ks'] = sample_['src_Ks']
                sample['ref_hw'] = sample_['ref_hw']
                sample['src_hw'] = sample_['src_hw']
                sample['near'] = sample_['near'].unsqueeze(0)
                sample['far'] = sample_['far'].unsqueeze(0)
            ref_rgb.append(sample_['ref_rgb'])
            ref_depth.append(sample_['ref_depth'])
            ref_w2c.append(sample_['ref_w2c'])
            ref_c2w.append(sample_['ref_c2w'])
            ref_K.append(sample_['ref_K'])
            ref_view_id.append(sample_['ref_view_id'])
            valid_mask.append(sample_['valid_mask'])
        sample['ref_rgb'] = torch.stack(ref_rgb, dim=0)
        sample['ref_depth'] = torch.stack(ref_depth, dim=0)
        sample['ref_w2c'] = torch.stack(ref_w2c, dim=0)
        sample['ref_c2w'] = torch.stack(ref_c2w, dim=0)
        sample['ref_K'] = torch.stack(ref_K, dim=0)
        sample['ref_view_id'] = torch.tensor(ref_view_id)
        sample['valid_mask'] = torch.stack(valid_mask, dim=0)
        return sample
    def __len__(self):
        return len(self.metas)
    
if __name__ == '__main__':
    from utils.config import parser
    args = parser.parse_args()
    path = '/home/jiahao/nerf/code/VGNF/RNVSG/data/DTU'
    # scenes = [scn for scn in os.listdir(path) if scn[:4] == 'scan']
    scenes = ['scan118']
    dtu = DTU(args, path, scenes, stage='train', num_images=64)
    data = dtu[0]
    for k, v in data.items():
        try:
            print(k, v.shape)
        except:
            print(k, v)
