import sys, os; sys.path.append(os.getcwd()); os.environ['CUDA_VISIBLE_DEVICES'] = '1'
try:
    sys.path.remove('/home/jiahao/nerf/code/VGNF/epcq/epcq_grid')
except:
    pass
import time
import torch, tqdm
from torch import nn
import matplotlib.pyplot as plt
# from pytorch_lightning.utilities import move_data_to_device
def move_data_to_device(batch, device):
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


from utils.utils import printTime
from epcq.epcq import NeuralPoints, CameraSpec, RenderOptions, FeatureQuery, Aggregator, render, clamp_t

class EPCQ(nn.Module):
    def __init__(self, ckpt_p=None, pointnerf_path=None, gt_pc_path=None, pc_type='pointnerf'):
        super(EPCQ, self).__init__()
        self.neural_points = NeuralPoints(pointnerf_path=pointnerf_path, gt_pc_path=gt_pc_path, pc_type=pc_type) # TODO: add arguments
        self.options = RenderOptions()
        self.feature_query = FeatureQuery()
        self.aggregator = Aggregator()
        self._load_ckpt(ckpt_p)

    def _load_ckpt(self, ckpt):
        if ckpt is None:
            return 
        if isinstance(ckpt, str):
            ckpt = torch.load(ckpt)
        for key in ckpt.keys():
            if 'xyz' in key:
                self.neural_points.xyz_data.data = ckpt[key]
            if 'rgb' in key:
                self.neural_points.rgb_data.data = ckpt[key]
            if 'color' in key:
                self.neural_points.rgb_data.data = ckpt[key].squeeze(0)
            if 'dir' in key:
                self.neural_points.dir_data.data = ckpt[key].squeeze(0)
            if 'feat' in key:
                self.neural_points.feat_data.data = ckpt[key]
            if 'points_embedding' in key:
                self.neural_points.feat_data.data = ckpt[key].squeeze(0)
            if 'density' in key:
                self.neural_points.density_data.data = ckpt[key]
            if 'conf' in key:
                self.neural_points.density_data.data = ckpt[key].squeeze(0)

        for k in self.aggregator.state_dict():
            self.aggregator.state_dict()[k].copy_(ckpt['aggregator.'+k])
            
        print("load ckpt successfully!")

    def forward(self, data, print_time = True):
        timeDict = dict()
        t0 = time.time()
        fx, fy, cx, cy = CameraSpec.decomposeK(data['ref_K'])
        cam = CameraSpec(c2w=data['ref_c2w'],
                         w2c=data['ref_w2c'],
                         K = data['ref_K'],
                         fx=fx, fy=fy,
                         cx=cx, cy=cy,
                         height=data['ref_hw'][0],
                         width=data['ref_hw'][1])

        sorted_point_idx_list, sorted_depth_list, pc_hash_table = self.neural_points.pts2plane(cam, self.options, sort=False)
        torch.cuda.synchronize()    
        t1 = time.time()
        timeDict['rasterization'] = t1 - t0
        
        output_sp_xyz, output_pc_idx, output_sp_t, mask_out = self.neural_points.render_image_hashtable_topk(cam, 
                                                                            self.options, 
                                                                            sorted_point_idx_list, 
                                                                            sorted_depth_list, 
                                                                            pc_hash_table, 
                                                                            save_variables_for_debug=False)
        
        if len(data['training_ray_index']) != 0:
            ray_index_v = data['training_ray_index'][:, 1]
            ray_index_u = data['training_ray_index'][:, 0]
            output_sp_xyz = output_sp_xyz[ray_index_v, ray_index_u, :, :]
            output_pc_idx = output_pc_idx[ray_index_v, ray_index_u, :, :]
            output_sp_t = output_sp_t[ray_index_v, ray_index_u, :]
            mask_out = mask_out[ray_index_v, ray_index_u, :]

        torch.cuda.synchronize()    
        t2 = time.time()
        timeDict['sampling'] = t2 - t1

        pc_xyz, pc_rgb, pc_dir, pc_feat, output_sp_xyz, valid_sp_mask_out, rays_d = self.feature_query(self.neural_points, output_sp_xyz, output_pc_idx, mask_out, cam, data['training_ray_index'])
        torch.cuda.synchronize()    
        t3 = time.time()
        timeDict['feature_query'] = t3 - t2

        final_alphas_pred, final_colors_pred = self.aggregator(pc_xyz, pc_rgb, pc_dir, pc_feat, output_sp_xyz, valid_sp_mask_out, mask_out, rays_d)
        torch.cuda.synchronize()
        t4 = time.time()
        timeDict['aggregation'] = t4 - t3

        if len(output_sp_t.shape) == 3:
            output_sp_t = output_sp_t.contiguous().view(cam.height*cam.width, -1)
        rgb_final = render(clamp_t(output_sp_t),
                           final_alphas_pred, 
                           final_colors_pred,
                           white_bg=True
                           )
        torch.cuda.synchronize()
        t5 = time.time()
        timeDict['rendering'] = t5 - t4
        if print_time:
            printTime(timeDict)
        return rgb_final

    
if __name__ == "__main__":
    from dataset.nerfsynthesis import NeRFSynthesis
    import numpy as np
    from utils.config import parser
    SCENE = 'lego'
    parser.add_argument('--scene', default = SCENE, dest='scene', type=str)
    
    img_index = 0#66
    args = parser.parse_args()
    status = 'inference'#'train_on_pc'
    pc_type = 'point_nerf' #'point_nerf' #'gt_pc'
    nerfsynthesis = NeRFSynthesis(args, args.scene, 
                                  ref_downsample=1,
                                  src_downsample=1,
                                  split='test',
                                  status=status,
                                  pc_type=pc_type)
    data = move_data_to_device(nerfsynthesis[img_index], torch.device(args.device))
    
    if nerfsynthesis.pc_type == 'point_nerf':
        model = EPCQ(ckpt_p=os.path.join(r'pointnerf_ckpt/ckpt', args.scene + '.pth'), pointnerf_path=os.path.join(r'pointnerf_ckpt/ckpt', args.scene + '.pth')).to(device=args.device)
    elif nerfsynthesis.pc_type == 'gt_pc':
        model = EPCQ(gt_pc_path=nerfsynthesis.pc_path,\
                    pc_type=nerfsynthesis.pc_type).to(device=args.device)

    repeate_ = 8
    font_size = 20
    with torch.no_grad():
        for i in range(repeate_):
            pred_rgb = model(data, print_time=True)
            # pred_rgb = pred_rgb.clamp_(0, 1)
            # pred_rgb = pred_rgb.cpu().numpy().reshape(800, 800, 3)
            # # depth_out = depth_out.cpu().numpy()
            # ref_img = data['ref_rgb'].cpu().numpy()
            # mse = ((pred_rgb - ref_img) ** 2).mean()
            # psnr = -10 * np.log10(mse)
            # fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            # axes = axes.flatten()
            # axes[0].imshow(ref_img)
            # axes[0].text(50, 50, 'ref_img', fontdict={'size': font_size})
            # axes[0].scatter(x=450, y=165, c='b', s=20)
            # axes[0].axis('off')
            # axes[1].imshow(pred_rgb)
            # axes[1].text(50, 50, f'pred_rgb psnr:{psnr:.2f}', fontdict={'size': font_size})
            # axes[1].axis('off')
            # # axes[2].imshow(depth_out)
            # # axes[2].text(50, 50, f'depth_out', fontdict={'size': font_size})
            # # axes[2].axis('off')
            # plt.tight_layout(pad=0, w_pad=0, h_pad=0)
            # plt.show()
            # # plt.savefig('cuda.png')
            # print('save image successfully!')