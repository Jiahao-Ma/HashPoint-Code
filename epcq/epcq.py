import torch, os
from dataclasses import dataclass
import torch.nn as nn
from typing import Union, NamedTuple, Optional, Tuple
from functools import reduce
import open3d as o3d


from . import utils
_C = utils._get_c_extension()

# debug 
import numpy as np

@dataclass
class CameraSpec:
    c2w:    torch.Tensor
    w2c:    torch.Tensor
    K  :    torch.Tensor
    fx :    float
    fy :    float
    cx :    float
    cy :    float
    width:  int
    height: int

    def _to_cpp(self, ):
        spec = _C.CameraSpec()
        spec.c2w = self.c2w
        spec.w2c = self.w2c
        spec.fx = self.fx
        spec.fy = self.fy
        spec.cx = self.cx
        spec.cy = self.cy
        spec.width = self.width
        spec.height = self.height
        return spec
    
    @staticmethod
    def decomposeK(K:torch.Tensor):
        return K[0, 0], K[1, 1], K[0, 2], K[1, 2] # fx, fy, cx, cy

    def _save2torch(self, filename):
        data = {
            'c2w': self.c2w,
            'w2c': self.w2c,
            'fx' : torch.tensor(self.fx.clone().detach()),
            'fy' : torch.tensor(self.fy.clone().detach()),
            'cx' : torch.tensor(self.cx.clone().detach()),
            'cy' : torch.tensor(self.cy.clone().detach()),
            'width': torch.tensor(self.width.clone().detach()),
            'height': torch.tensor(self.height.clone().detach()),
        }
        container = torch.jit.script(utils.Container(data))
        container.save(filename)
        print('save to', filename)
    
    
    def pixelidx2hwidx(self, pixelidx):
        return pixelidx // self.width, pixelidx % self.width # return h, w
    
    def hwidx2pixelidx(self, h_idx, w_idx):
        return h_idx * self.width + w_idx
    
# @dataclass
# class RenderOptions:
#     kernel_size: int = 5
#     t_intvl: float = 0.01 # 0.02
#     radius_threshold: float = 0.016
#     sigma_threshold: float = 1e-4
#     background_brightness: float = 1.0
#     stop_threshold: float = 0.001
#     num_sample_points_per_ray: int = 1 #2 #4 #8 #16
#     num_point_cloud_per_sp: int = 8
#     num_point_per_ray: int = 256
#     sdf2weight_gaussian_alpha: float = 0.002#0.002 # the parameter for GAUSSIAN_WEIGHT

#     sparsity_loss_w: float = 5e-4
    
#     topK: int = 8 # the number of nearest points for each sample point
@dataclass
class RenderOptions:
    kernel_size: int = 5
    t_intvl: float = 0.01 # 0.02
    radius_threshold: float = 0.016
    sigma_threshold: float = 1e-4
    background_brightness: float = 1.0
    stop_threshold: float = 0.001
    num_sample_points_per_ray: int = 16 #4 #8 #16
    num_point_cloud_per_sp: int = 4
    num_point_per_ray: int = 256
    sdf2weight_gaussian_alpha: float = 0.2 #0.2 #0.002 # the parameter for GAUSSIAN_WEIGHT
    sdf2weight_gaussian_gamma: float = 0.9 # ranged from 1 to 0
    sparsity_loss_w: float = 5e-4
    
    topK: int = 8 
    def _to_cpp(self, ):
        opt = _C.RenderOptions()
        opt.kernel_size = self.kernel_size
        opt.t_intvl = self.t_intvl
        opt.radius_threshold = self.radius_threshold
        opt.sigma_threshold = self.sigma_threshold
        opt.background_brightness = self.background_brightness
        opt.stop_threshold = self.stop_threshold
        opt.num_sample_points_per_ray = self.num_sample_points_per_ray
        opt.num_point_cloud_per_sp = self.num_point_cloud_per_sp
        opt.num_point_per_ray = self.num_point_per_ray
        # opt.RasterizeStrategies = self.RasterizeStrategies
        # opt.RenderStrategies = self.RenderStrategies
        # opt.sdf2weight_type = self.sdf2weight_type
        opt.sdf2weight_gaussian_alpha = self.sdf2weight_gaussian_alpha
        opt.sdf2weight_gaussian_gamma = self.sdf2weight_gaussian_gamma
        opt.topK = self.topK
        return opt
    
    def _save2torch(self, filename):
        data = {
            'kernel_size': torch.tensor(self.kernel_size),
            't_intvl': torch.tensor(self.t_intvl),
            'radius_threshold': torch.tensor(self.radius_threshold),
            'sigma_threshold': torch.tensor(self.sigma_threshold),
            'background_brightness': torch.tensor(self.background_brightness),
            'stop_threshold': torch.tensor(self.stop_threshold),
            'num_sample_points_per_ray': torch.tensor(self.num_sample_points_per_ray),
            'num_point_cloud_per_sp': torch.tensor(self.num_point_cloud_per_sp),
            'sdf2weight_gaussian_alpha': torch.tensor(self.sdf2weight_gaussian_alpha),
            'sdf2weight_gaussian_gamma': torch.tensor(self.sdf2weight_gaussian_gamma),
        }
        container = torch.jit.script(utils.Container(data))
        container.save(filename)
        print('save to', filename)
    
@dataclass
class NeuralPointsGrads:
    grad_sigma: torch.Tensor
    grad_sh: torch.Tensor
    mask_out: torch.Tensor
    
    def _to_cpp(self, ):
        grads = _C.NeuralPointsGrads()
        grads.grad_sigma_out = self.grad_sigma
        grads.grad_sh_out = self.grad_sh
        grads.mask_out = self.mask_out
        return grads
    

class NeuralPoints(nn.Module):
    def __init__(self, 
                 basis_dim: int = 9, # deg3: 1 + 3 + 5    
                 device: Union[torch.device, str] = "cuda:0",
                 pc_type: str='pointnerf', # gt_pc or pointnerf or pc
                 gt_pc_path: str='/home/jiahao/nerf/data/nerfs/nerf_sythetic/lego/point_clouds/pointclouds.ply',
                 pointnerf_path: str='pointnerf_ckpt/nerf_synthesis/lego.pth',
                 alpha: float=5.0,
                 points = None,
                 ):
        super(NeuralPoints, self).__init__()
        assert pc_type in ['pointnerf', 'gt_pc', 'pc']
        self.pc_type = pc_type 
        if pc_type == 'pointnerf':
            if pointnerf_path.endswith('.pth'):
                ckpt = torch.load(pointnerf_path)
                self.xyz_data = ckpt['neural_points.xyz']            
                self.rgb_data = ckpt['neural_points.points_color'].squeeze(0)
                self.dir_data = ckpt['neural_points.points_dir'].squeeze(0)
                self.density_data =ckpt['neural_points.points_conf'].squeeze(0)
                self.feat_data =  ckpt['neural_points.points_embeding'].squeeze(0)
                
                self.xyz_data = nn.Parameter(self.xyz_data, requires_grad=False)
                self.rgb_data = nn.Parameter(self.rgb_data, requires_grad=True)
                self.dir_data = nn.Parameter(self.dir_data, requires_grad=True)
                self.density_data = nn.Parameter(self.density_data, requires_grad=True)
                self.feat_data = nn.Parameter(self.feat_data, requires_grad=True)

                self.capacity = self.xyz_data.shape[0]
            else:
                raise ValueError("Unsupported pointnerf_path: {}".format(pointnerf_path))
        elif pc_type == 'gt_pc':
            if gt_pc_path is None:
                raise ValueError('gt_pc_path is required')
            # load pc using o3d
            pcd = o3d.io.read_point_cloud(gt_pc_path)
            self.xyz_data = nn.Parameter(
                torch.tensor(np.array(pcd.points), dtype=torch.float32, device=device),
                requires_grad=False,
            )
            self.rgb_data = nn.Parameter(
                torch.tensor(np.array(pcd.colors), dtype=torch.float32, device=device),
                requires_grad=True,
            )
            self.dir_data = nn.Parameter(
                torch.zeros_like(self.xyz_data), requires_grad=True,
            )

            self.capacity = self.xyz_data.shape[0]

            self.basis_dim = basis_dim
            self.density_data = nn.Parameter(
                torch.zeros(self.capacity, 1, dtype=torch.float32, device=device),
                requires_grad=True,
            )
            
            self.feat_data = nn.Parameter(
                torch.zeros(
                    self.capacity, 32, dtype=torch.float32, device=device),
                requires_grad=True,
            )

        elif pc_type == 'pc':
            assert points is not None
            # random init
            self.xyz_data = nn.Parameter(points, requires_grad=False)
            self.rgb_data = nn.Parameter(torch.zeros_like(points), requires_grad=False)
            self.dir_data = nn.Parameter(torch.zeros_like(points), requires_grad=False)
            self.density_data = nn.Parameter(torch.zeros((points.shape[0], 1)), requires_grad=False)
            self.feat_data = nn.Parameter(torch.zeros_like(points), requires_grad=False)
            self.capacity = self.xyz_data.shape[0]
            
        self.multi_bins_rasterize = _C.__dict__['scatter']
        self.rasterize_hashtable = _C.__dict__['scatter_hashtable']
        self.quick_sampling = _C.__dict__['quick_sampling']
       
    def _load_ckpt(self, ckpt):
        if self.pc_type == 'gt_pc':
            if isinstance(ckpt, str):
                ckpt = torch.load(ckpt)
            self.density_data.data = ckpt['density_data']
            self.sh_data.data = ckpt['sh_data']
            self.xyz_data.data = ckpt['xyz_data']
            if 'alpha_data' in ckpt.keys():
                self.alpha_data.data = ckpt['alpha_data']
        else:
            raise NotImplementedError
                
    def _to_cpp(self, ):
        spec = _C.NeuralPointsSpec()
        spec.xyz_data = self.xyz_data 
        return spec
    
    def _save2torch(self, filename):
        data = {
            'density_data': self.density_data,
            'sh_data': self.feat_data,
            'xyz_data': self.xyz_data,
            'basis_dim': torch.tensor(9),
        }
        container = torch.jit.script(utils.Container(data))
        container.save(filename)
        print('save to', filename)
        
    def _world2cam(self, cam:CameraSpec, pc_xyz:torch.Tensor=None):
        if pc_xyz is None:
            pts_c = (cam.w2c[:3, :3] @ self.xyz_data.t() + cam.w2c[:3, 3:4]).t() # [N, 3]
        else:
            pts_c = (cam.w2c[:3, :3] @ pc_xyz.t() + cam.w2c[:3, 3:4]).t()
        depth = pts_c[:, 2:3]
        return pts_c, depth
    
    def _cam2image(self, pts_c:torch.Tensor,
                   cam:CameraSpec):
        pts_scrn = cam.K @ pts_c.t() # [3, 3] @ [3, N] -> [3, N]
        pts_scrn = pts_scrn / pts_scrn[2:3, :] # [3, N] / [1, N] -> [3, N]
        screen_x = pts_scrn[0, :]
        screen_y = pts_scrn[1, :]
        return screen_x, screen_y, screen_x.to(torch.int32), screen_y.to(torch.int32)
    
    def _convetpixel(self, screen_x, screen_y, hw):
        # convert pixel to image
        point_idx = screen_x + screen_y * hw[1]
        out_of_bound_mask = ((0<=screen_x) & (screen_x<hw[1]) & (0<=screen_y) & (screen_y<hw[0]))
        point_idx = torch.where(out_of_bound_mask,
                                point_idx,
                                torch.full_like(point_idx, hw[0]*hw[1]))
        return point_idx

    def _uv2ray_index(self, uv, width):
        return (uv[:, 1] * width + uv[:, 0]).to(torch.int32)

    def pts2plane(self, cam:CameraSpec, opt:RenderOptions, pc_xyz:torch.Tensor=None, sort:bool=False):
        pts_c, depth = self._world2cam(cam, pc_xyz)
        _, _, screen_x_d, screen_y_d = self._cam2image(pts_c, cam)
        point2img_idx = self._convetpixel(screen_x_d, screen_y_d, [cam.height, cam.width])
        point_idx = torch.arange(self.capacity, dtype=torch.int32, device=self.xyz_data.device)
        point2img_idx, sorted_index = torch.sort(point2img_idx.squeeze(), dim=-1, descending=False)
        depth = depth.squeeze(1)[sorted_index]
        point_idx = point_idx[sorted_index]
      
        sorted_point_idx_list, sorted_depth_list, hash_table = self.rasterize_hashtable(point_idx,
                                point2img_idx,
                                depth,
                                [cam.height, cam.width],
                                sort
                                )
        
        return sorted_point_idx_list, sorted_depth_list, hash_table

    def render_image_hashtable_topk(self, cam:CameraSpec, 
                                    opt:RenderOptions,
                                    sorted_point_idx_list:torch.Tensor,
                                    sorted_depth_list:torch.Tensor,
                                    hashtable:torch.Tensor,
                                    save_variables_for_debug:bool=False):
        if save_variables_for_debug:
            sorted_point_idx_list_ = {'sorted_point_idx_list' : sorted_point_idx_list}
            sorted_point_idx_list_ctn = torch.jit.script(utils.Container(sorted_point_idx_list_))
            sorted_point_idx_list_ctn.save(r'/home/jiahao/nerf/code/VGNF/epcq/cuda_debug_epcq_grid1/variables1/sorted_point_idx_list.pt')
            
            sorted_depth_list_ = {'sorted_depth_list' : sorted_depth_list}
            sorted_depth_list_ctn = torch.jit.script(utils.Container(sorted_depth_list_))
            sorted_depth_list_ctn.save(r'/home/jiahao/nerf/code/VGNF/epcq/cuda_debug_epcq_grid1/variables1/sorted_depth_list.pt')
            
            hashtable_ = {'hashtable' : hashtable}
            hashtable_ctn = torch.jit.script(utils.Container(hashtable_))
            hashtable_ctn.save(r'/home/jiahao/nerf/code/VGNF/epcq/cuda_debug_epcq_grid1/variables1/hashtable.pt')
            
            self._save2torch(r'/home/jiahao/nerf/code/VGNF/epcq/cuda_debug_epcq_grid1/variables1/epcq.pt')
            cam._save2torch(r'/home/jiahao/nerf/code/VGNF/epcq/cuda_debug_epcq_grid1/variables1/cam.pt')
            opt._save2torch(r'/home/jiahao/nerf/code/VGNF/epcq/cuda_debug_epcq_grid1/variables1/opt.pt')
        
        output_sp_xyz, output_pc_idx, output_sp_t, mask_out = self.quick_sampling(self._to_cpp(),
                                    cam._to_cpp(),
                                    opt._to_cpp(),
                                    sorted_point_idx_list,
                                    hashtable
                                    )
        
        return output_sp_xyz, output_pc_idx, output_sp_t, mask_out
    
class FeatureQuery(nn.Module):
    def __init__(self) -> None:
        super(FeatureQuery, self).__init__()
        pass
    
    def generate_rays(self, cam:CameraSpec):
        x,y = torch.meshgrid([torch.arange(cam.width, dtype=torch.float32, device=cam.fx.device),
                                torch.arange(cam.height, dtype=torch.float32, device=cam.fx.device)],
                                indexing='xy',
                                )
        rays = torch.stack([ (x - cam.cx) / cam.fx, (y - cam.cy) / cam.fy, torch.ones_like(x)], dim=-1)
        rays_d = torch.sum(rays[:, :, None, :] * cam.c2w[:3, :3][None, None, :, :], dim=-1)
        rays_o = torch.Tensor(cam.c2w[:3, 3]).expand_as(rays_d)
        return rays_o, rays_d


    def forward(self, neuralpoints:NeuralPoints, output_sp_xyz:torch.Tensor, output_pc_idx:torch.Tensor, mask_out:torch.Tensor, cam:CameraSpec, ray_index_uv:torch.Tensor=None):
        num_sp_per_ray, num_pc_per_sp = output_pc_idx.shape[-2:]
        valid_ray_mask_out = mask_out.flatten() # [h, w, num_sp_per_ray] -> [h*w*num_sp_per_ray]
        output_sp_xyz = output_sp_xyz.contiguous().view(-1, 3)
        output_pc_idx = output_pc_idx.contiguous().view(-1, num_pc_per_sp).long()
        output_sp_xyz = output_sp_xyz[valid_ray_mask_out]
        output_pc_idx = output_pc_idx[valid_ray_mask_out]
        num_valid_ray = output_pc_idx.shape[0]
        output_pc_idx = output_pc_idx.flatten()
        valid_sp_mask_out = (output_pc_idx != -1)
        output_pc_idx = output_pc_idx[valid_sp_mask_out]

        pc_xyz_ = torch.gather(neuralpoints.xyz_data, dim=0, index=output_pc_idx.unsqueeze(-1).expand(-1, 3))
        pc_rgb_ = torch.gather(neuralpoints.rgb_data, dim=0, index=output_pc_idx.unsqueeze(-1).expand(-1, 3))
        pc_dir_ = torch.gather(neuralpoints.dir_data, dim=0, index=output_pc_idx.unsqueeze(-1).expand(-1, 3))
        pc_feat_ = torch.gather(neuralpoints.feat_data, dim=0, index=output_pc_idx.unsqueeze(-1).expand(-1, neuralpoints.feat_data.shape[-1]))

        pc_xyz = torch.zeros((num_valid_ray*num_pc_per_sp, 3), device=neuralpoints.xyz_data.device)
        pc_xyz[valid_sp_mask_out] = pc_xyz_
        pc_xyz = pc_xyz.contiguous().view(num_valid_ray, num_pc_per_sp, 3)

        pc_rgb = torch.zeros((num_valid_ray*num_pc_per_sp, 3), device=neuralpoints.rgb_data.device)
        pc_rgb[valid_sp_mask_out] = pc_rgb_
        pc_rgb = pc_rgb.contiguous().view(num_valid_ray, num_pc_per_sp, 3)

        pc_dir = torch.zeros((num_valid_ray*num_pc_per_sp, 3), device=neuralpoints.dir_data.device)
        pc_dir[valid_sp_mask_out] = pc_dir_
        pc_dir = pc_dir.contiguous().view(num_valid_ray, num_pc_per_sp, 3)

        pc_feat = torch.zeros((num_valid_ray*num_pc_per_sp, pc_feat_.shape[-1]), device=neuralpoints.feat_data.device)
        pc_feat[valid_sp_mask_out] = pc_feat_
        pc_feat = pc_feat.contiguous().view(num_valid_ray, num_pc_per_sp, pc_feat_.shape[-1])

        valid_sp_mask_out = valid_sp_mask_out.contiguous().view(num_valid_ray, num_pc_per_sp)

        _, rays_d = self.generate_rays(cam)
        if len(ray_index_uv) != 0:
            rays_d = rays_d[ray_index_uv[:, 1], ray_index_uv[:, 0]]
            rays_d = rays_d[:, None, :].expand(-1, num_sp_per_ray, -1).contiguous().view(-1, 3)
        else:
            rays_d = rays_d[:, :, None, :].expand(-1, -1, num_sp_per_ray, -1).contiguous().view(-1, 3)
        rays_d = rays_d[valid_ray_mask_out]
        return pc_xyz, pc_rgb, pc_dir, pc_feat, output_sp_xyz, valid_sp_mask_out, rays_d

def positional_encoding(positions, freqs, ori=False):
    '''encode positions with positional encoding
        positions: :math:`(...,D)`
        freqs: int
    Return:
        pts: :math:`(..., 2DF)`
    '''
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    ori_c = positions.shape[-1]
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))  # (..., DF)
    if ori:
        pts = torch.cat([positions, torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2+ori_c,))
    else:
        pts = torch.stack([torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2,))
    return pts

class Aggregator(nn.Module):
    def __init__(self) -> None:
        super(Aggregator, self).__init__()
        self.block1 = nn.Sequential(nn.Linear(in_features=284, out_features=256, bias=True), nn.LeakyReLU(inplace=True),
                                    nn.Linear(in_features=256, out_features=256, bias=True), nn.LeakyReLU(inplace=True),
                                    )
        
        self.block3 = nn.Sequential(nn.Linear(in_features=263, out_features=256, bias=True), nn.LeakyReLU(inplace=True),
                                    nn.Linear(in_features=256, out_features=256, bias=True), nn.LeakyReLU(inplace=True),
                                    )
        
        self.alpha_branch = nn.Sequential(nn.Linear(in_features=256, out_features=1))
        self.alpha_act = torch.nn.Softplus()
        self.color_branch = nn.Sequential(nn.Linear(in_features=280, out_features=128, bias=True), nn.LeakyReLU(inplace=True),
                                          nn.Linear(in_features=128, out_features=128, bias=True), nn.LeakyReLU(inplace=True),
                                          nn.Linear(in_features=128, out_features=128, bias=True), nn.LeakyReLU(inplace=True),
                                          nn.Linear(in_features=128, out_features=3, bias=True)
                                          )
        self.color_act = torch.nn.Sigmoid()
        # initialization with xavier_uniform + bias init 0.0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        self.chunksize = 20000

    def calculate_weight(self, dist, mask=None):
        if mask is not None:
            return ((1 / (dist+1e-6)) * mask.float())/ ((torch.sum( (1 / (dist+1e-6) * mask.float()), dim=-1, keepdim=True) + 1e-6))
        return (1 / (dist+1e-6)) / (torch.sum( (1 / (dist+1e-6)), dim=-1, keepdim=True) + 1e-6)

    def forward(self, pc_xyz, pc_rgb, pc_dir, pc_feat, output_sp_xyz, valid_sp_mask_out, mask_out, rays_d):
        alphas_list = list()
        colors_list = list()
        for idx, i in enumerate(range(0, valid_sp_mask_out.shape[0], self.chunksize)):
            valid_sp_mask_ = valid_sp_mask_out[i:i+self.chunksize]
            pc_xyz_ = pc_xyz[i:i+self.chunksize]
            pc_rgb_ = pc_rgb[i:i+self.chunksize].contiguous().view(-1, 3)[valid_sp_mask_.flatten()]
            pc_dir_ = pc_dir[i:i+self.chunksize].contiguous().view(-1, 3)[valid_sp_mask_.flatten()]
            pc_feat_ = pc_feat[i:i+self.chunksize]
            sp_xyz_ = output_sp_xyz[i:i+self.chunksize]
            dist = pc_xyz_ - sp_xyz_.unsqueeze(1)
            distpow = pc_xyz_.pow(2) - sp_xyz_.unsqueeze(1).pow(2)
            dist_feat = torch.cat([dist, distpow], dim=-1)
            weights = self.calculate_weight(torch.norm(dist, dim=-1), valid_sp_mask_)

            dist_feat = dist_feat.contiguous().view(-1, 6)[valid_sp_mask_.flatten()]
            embedded_dist_feat = positional_encoding(dist_feat, 5, False)
            
            pc_feat_ = pc_feat_.contiguous().view(-1, pc_feat_.shape[-1])[valid_sp_mask_.flatten()]
            embedded_pc_feat = positional_encoding(pc_feat_, 3, False)

            rays_d_ = rays_d[i:i+self.chunksize][:, None, :].expand(-1, pc_xyz_.shape[1], -1).contiguous().view(-1, 3)[valid_sp_mask_.flatten()]
            embedded_ray_dir = positional_encoding(rays_d[i:i+self.chunksize], 4, False)

            feat = torch.cat([pc_feat_, embedded_pc_feat, embedded_dist_feat], dim=-1)
            feat = self.block1(feat)

            feat = torch.cat([feat, pc_rgb_, pc_dir_ - rays_d_, torch.sum(pc_dir_ * rays_d_, dim=-1, keepdim=True)], dim=-1)
            feat = self.block3(feat)

            alphas = self.alpha_branch(feat)
            alphas = self.alpha_act(alphas - 1)

            alpha_pred = torch.zeros((valid_sp_mask_.shape[0]*valid_sp_mask_.shape[1], 1), dtype=torch.float32, device=valid_sp_mask_.device)
            alpha_pred[valid_sp_mask_.flatten(), :] = alphas
            alpha_pred = alpha_pred.contiguous().view(valid_sp_mask_.shape[0], valid_sp_mask_.shape[1])
            alpha_pred = torch.sum(alpha_pred * weights, dim=-1, keepdim=True)
            alphas_list.append(alpha_pred)

            feat_canvas = torch.zeros((valid_sp_mask_.shape[0]*valid_sp_mask_.shape[1], feat.shape[1]), device=feat.device, dtype=feat.dtype)
            feat_canvas[valid_sp_mask_.flatten(), :] = feat
            feat_canvas = feat_canvas.contiguous().view(valid_sp_mask_.shape[0], valid_sp_mask_.shape[1], feat.shape[1])
            feat_canvas = torch.sum(feat_canvas * weights.unsqueeze(-1), dim=-2)
            feat = torch.cat([feat_canvas, embedded_ray_dir], dim=-1)

            color_pred = self.color_act(self.color_branch(feat))
            color_pred = color_pred * (1 + 2 * 0.001) - 0.001
            colors_list.append(color_pred)


        colors_list = torch.cat(colors_list, dim=0)
        alphas_list = torch.cat(alphas_list, dim=0)
        if len(mask_out.shape) == 3: # inference
            final_alphas_pred = torch.zeros((mask_out.shape[0] * mask_out.shape[1] * mask_out.shape[2]), device=mask_out.device)
            final_alphas_pred[mask_out.flatten()] = alphas_list.flatten()

            final_colors_pred = torch.zeros((mask_out.shape[0] * mask_out.shape[1] * mask_out.shape[2], 3), device=mask_out.device)
            final_colors_pred[mask_out.flatten(), :] = colors_list
            return final_alphas_pred.contiguous().view(mask_out.shape[0] * mask_out.shape[1], mask_out.shape[2]), \
                final_colors_pred.contiguous().view(mask_out.shape[0] * mask_out.shape[1], mask_out.shape[2], 3)
        elif len(mask_out.shape) == 2: # training
            final_alphas_pred = torch.zeros((mask_out.shape[0] * mask_out.shape[1]), device=mask_out.device)
            final_alphas_pred[mask_out.flatten()] = alphas_list.flatten()
            
            final_colors_pred = torch.zeros((mask_out.shape[0] * mask_out.shape[1], 3), device=mask_out.device)
            final_colors_pred[mask_out.flatten(), :] = colors_list
            return final_alphas_pred.contiguous().view(mask_out.shape[0], mask_out.shape[1]), final_colors_pred.contiguous().view(mask_out.shape[0], mask_out.shape[1], 3)
        else:
            raise ValueError('mask_out shape error')
def clamp_t(t_val, vsize=0.004):
    t_dist = t_val[:, 1:] - t_val[:, :-1]
    t_dist = torch.cat([t_dist, torch.full_like(t_dist[:, :1], vsize)], dim=-1)
    mask = torch.logical_or(t_dist < 1e-8, t_dist > 2 * vsize)
    mask = mask.to(torch.float32)
    t_dist = t_dist * (1.0 - mask) + mask * vsize

    return t_dist

def render(deltas, sigmas, rgbs, white_bg=False):
    opacity = 1 - torch.exp( - sigmas * deltas)
    acc_transmission = torch.cumprod(1. - opacity + 1e-10, dim=-1)
    temp = torch.ones((opacity.shape[0], 1), dtype=torch.float32, device=opacity.device)
    acc_transmission = torch.cat([temp, acc_transmission[..., :-1]], dim=-1)
    blend_weight = opacity * acc_transmission
    ray_color = torch.sum(blend_weight.unsqueeze(-1) * rgbs, dim=-2)
    if white_bg:
        bg_color = torch.ones_like(ray_color)
    else:
        bg_color = torch.zeros_like(ray_color) # black background by default
    background_transmission = acc_transmission[:, [-1]]        
    ray_color = ray_color + background_transmission * bg_color
    return ray_color