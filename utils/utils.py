import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch import nn
import time
from tqdm import tqdm
import open3d as o3d
def totensor(x):
    
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, Image.Image):
        return torch.from_numpy(np.array(x))
    else:
        raise NotImplementedError
    
def write_voxels(fn, verts, faces):
    with open(fn, 'w') as f:
        f.write(f'ply\n')
        f.write(f'format ascii 1.0\n')
        f.write(f'element vertex {len(verts)}\n')
        f.write(f'property float x\n')
        f.write(f'property float y\n')
        f.write(f'property float z\n')
        f.write(f'property uchar red\n')
        f.write(f'property uchar green\n')
        f.write(f'property uchar blue\n')
        f.write(f'property uchar alpha\n')
        f.write(f'element face {len(faces)}\n')
        f.write(f'property list uchar int vertex_indices\n')
        f.write(f'end_header\n')
        for v in verts:
            f.write(f'{v[0]} {v[1]} {v[2]} {int(v[3])} {int(v[4])} {int(v[5])} 0\n')
        for face in faces:
            f.write(f'3 {face[0]} {face[1]} {face[2]}\n')
            
def save_depth(depths): 
    import matplotlib.pyplot as plt
    for i, depth in enumerate(depths):
        depth = depth.squeeze()
        dmin, dmax = depth.min(), depth.max()
        depth = (depth - dmin) / (dmax - dmin)
        plt.imshow(depth.cpu().numpy())
        plt.axis('off')
        plt.savefig(f'visualization/depth_{i}.png', bbox_inches='tight',dpi=300, pad_inches=0.0)
        plt.close()
        
def write_point_clouds(ply_filename, points, to255=True, supplement_color=False, using_open3d=False):
    if not os.path.exists(os.path.dirname(ply_filename)):
        os.makedirs(os.path.dirname(ply_filename))
    formatted_points = []
    if supplement_color:
        rgbs = np.random.normal(size=points.shape)
        points = np.concatenate([points, rgbs], axis=-1)
    if using_open3d:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(ply_filename, pcd)
    else:
        if points.shape[-1] == 6:
            if to255:
                points[..., 3:6] = np.array(points[..., 3:6] * 255).astype(np.uint8)
            for point in tqdm(points):
                formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3], point[4], point[5])) # blue <-> red for CloudCompare visualization

            out_file = open(ply_filename, "w")
            out_file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            ''' % (len(points), "".join(formatted_points)))
            out_file.close()
        else:
            for point in tqdm(points):
                formatted_points.append("%f %f %f\n" % (point[0], point[1], point[2]))
            out_file = open(ply_filename, "w")
            out_file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            end_header
            %s
            ''' % (len(points), "".join(formatted_points)))
            out_file.close()
    
def export_voxel(voxel_feats, voxel_centers, voxel_len, pc_num_each_vxl, path='visualization/voxels.ply', fill_color=None):
    
    offsets = torch.tensor([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], 
                            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], \
                            device=voxel_centers.device) * voxel_len / 2.0 # (8, 3)
    vertex_pts = voxel_centers[:, None, :] + offsets[None, :, :]
    if fill_color is not None:
        colors = torch.ones_like(vertex_pts) * torch.Tensor(fill_color)[None, None, :]
        vertex_pts = torch.cat([vertex_pts, colors], dim=-1)
    else:
        # colors = torch.sum(voxel_feats[:, :, 3:6], dim=1) / pc_num_each_vxl[:, None] * 255
        colors = torch.max(voxel_feats[:, :, 3:6], dim=-2)[0] * 255
        # colors = voxel_feats[:, 0, 3:6] * 255
        colors = torch.clamp(colors, min=0., max=255.)
        colors = colors[:, None, :].repeat_interleave(8, dim=1)
        vertex_pts = torch.cat([vertex_pts, colors], dim=-1)
    faces = torch.tensor([[0,1,2],
                            [2,3,0],
                            [1,5,6],
                            [6,2,1],
                            [7,6,5],
                            [5,4,7],
                            [4,0,3],
                            [3,7,4],
                            [4,5,1],
                            [1,0,4],
                            [3,2,6],
                            [6,7,3]])
    all_faces = list()
    for i in range(vertex_pts.shape[0]):
        face_offset = i * len(offsets)
        cur_faces = faces + face_offset
        all_faces.append(cur_faces)
    vertex_faces = torch.cat(all_faces, dim=0)
    vertex_pts = vertex_pts.reshape(-1, 6)
    write_voxels(path, vertex_pts.numpy(), vertex_faces.numpy())
    
def get_random_color():
    r = np.random.uniform(0, 1)
    g = np.random.uniform(0, 1)
    b = np.random.uniform(0, 1)
    return np.array([r, g, b])

def vis_sample_points(pts3d, path='visualization/sample_points.ply', fill_color=None):
    all_pts = list()
    for pts in pts3d:
        if fill_color is not None:
            color = np.array(fill_color)[None, :]
        else:
            color = get_random_color()[None, :]
        colors = np.repeat(color, pts.shape[0], axis=0)
        pts = np.concatenate([pts, colors], axis=-1)
        all_pts.append(pts)
    all_pts = np.concatenate(all_pts)
    write_point_clouds(path, all_pts)
    
def vis_cage(center, radius):
    neg = center - radius
    pos = center + radius
    xyz = np.stack(np.meshgrid(np.linspace(start=neg[0], stop=pos[0], num=256),
                               np.linspace(start=neg[1], stop=pos[1], num=256),
                               np.linspace(start=neg[2], stop=pos[2], num=256),
                       ), axis=-1) # (256, 256, 256, 3)
    mask = np.ones((256, 256, 256))
    mask[1:-1, 1:-1, 1:-1] = 0
    mask = mask.astype(np.bool)
    xyz = xyz[mask]
    vis_sample_points(xyz[None, ...], path='visualization/cage.ply', fill_color=[1, 1, 0])
    
    
def save_img_depth(predrgb, gtrgb, depth, indx=0, psnr=None, root='visualization/images/'):
    if not os.path.exists(root):
        os.makedirs(root)
    
    plt.subplot(121)
    plt.imshow(gtrgb)
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(predrgb)
    plt.title(f'psnr: {psnr:.2f}', fontdict={'fontsize': 25})
    plt.axis('off')
    plt.savefig(os.path.join(root, f'rgb_{indx}.png'), bbox_inches='tight',dpi=300, pad_inches=0.0)
    plt.close()
    
    plt.imshow(depth)
    plt.axis('off')
    plt.savefig(os.path.join(root, f'depth_{indx}.png'), bbox_inches='tight',dpi=300, pad_inches=0.0)
    plt.close()

def save_gt_pred_img_depth_opencv(predrgb, gtrgb, depth, indx=0, psnr=0., root='visualization/images/'):
    '''
        `save_img_depth_opencv` is faster than `save_img_depth`
    '''
    if not os.path.exists(root):
        os.makedirs(root)
    depth = (depth[:, :, None].repeat(3, axis=-1)).astype(np.uint8)
    image = np.concatenate([gtrgb, predrgb, depth], axis=1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f'PSNR: {psnr:.2f}', (500, 30), font, 1, (0, 0, 0), 4, cv2.LINE_AA)
    image = image[..., ::-1]
    save_path = os.path.join(root, f'{indx}.png')
    cv2.imwrite(save_path, image)
    assert os.path.exists(save_path)
    
def save_img_depth_opencv(gtrgb, depth, indx=0, root='visualization/images/'):
    if not os.path.exists(root):
        os.makedirs(root)
    plt.subplot(121)
    plt.axis('off')
    depth = (depth[:, :, None].cpu().numpy()).squeeze()
    dmin, dmax = depth.min(), depth.max()
    depth = (depth - dmin) / (dmax - dmin)
    depth = np.clip( depth*255, a_min=0, a_max=255)
    depth = (depth[:, :, None].repeat(3, axis=-1)).astype(np.uint8)
    plt.imshow(depth)

    plt.subplot(122)
    plt.axis('off')
    gtrgb = (gtrgb.cpu().numpy() * 255).astype(np.uint8)
    plt.imshow(gtrgb)
    # plt.show()
    # image = np.concatenate([gtrgb, depth], axis=1)
    save_path = os.path.join(root, f'{indx}.png')
    plt.savefig(save_path, bbox_inches='tight',dpi=300, pad_inches=0.0)
    plt.close()
    # cv2.imwrite(save_path, image)
    # assert os.path.exists(save_path)
    print(f'{save_path} has been saved.')

def save_img(gt_img, pred_img, pred_depth=None, path=None, name=None):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    gt_img = (gt_img * 255).astype(np.uint8)
    pred_img = (pred_img * 255).astype(np.uint8)
    if pred_depth is not None:
        pred_depth = cv2.cvtColor(pred_depth, cv2.COLOR_GRAY2BGR)
        image = np.concatenate([gt_img, pred_img, pred_depth], axis=1)
    else:
        image = np.concatenate([gt_img, pred_img], axis=1)
    # write the name to the image
    cv2.putText(image, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.imwrite(path, image[..., ::-1])

def save_single_img(img, path=None, name=None):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, img[..., ::-1])

def normalize_depth(depth, depth_thresh=[-10, 10]):
    depth = np.clip(depth, a_min=depth_thresh[0], a_max=depth_thresh[1])
    dmin, dmax = depth.min(), depth.max()
    depth = (depth - dmin) / (dmax - dmin)
    depth = np.clip( depth*255, a_min=0, a_max=255)
    return depth.astype(np.uint8)

def imgs2gif(imgs, saveName, duration=0.06 * 10, loop=0, fps=None):
    if fps:
        duration = 1 / fps
    duration *= 1000
    
    imgs = [cv2.imread(img)[:, :, ::-1] for img in imgs]
    imgs = [Image.fromarray(img).convert('RGB') for img in imgs]
    imgs[0].save(saveName, save_all=True, append_images=imgs, duration=duration, loop=loop)

def color_map_backward(rgb):
    rgb = rgb * 255
    rgb = np.clip(rgb, a_min=0, a_max=255).astype(np.uint8)
    return rgb

def downsample_gaussian_blur(img, ratio):
    sigma = (1 / ratio) / 3
    # ksize=np.ceil(2*sigma)
    ksize = int(np.ceil(((sigma - 0.8) / 0.3 + 1) * 2 + 1))
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT101)
    return img

def downsample2ratio(downsample):
    return 1 / downsample

def resize_img(img, ratio):
    # if ratio>=1.0: return img
    h, w, _ = img.shape
    hn, wn = int(np.round(h * ratio)), int(np.round(w * ratio))
    img_out = cv2.resize(downsample_gaussian_blur(img, ratio), (wn, hn), cv2.INTER_LINEAR)
    return img_out

class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.xlim, self.ylim, self.zlim = xlim, ylim, zlim
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection = '3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()
    
    def reset(self):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection = '3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_zlim(self.zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')
        
def printTime(timeDict):
    colors = ['[32m', '[33m', '[35m', '[36m', '[37m', '[34m', '[31m']
    total_time = 0
    render_time = 0
    print("# ------------------- Time ------------------- #")
    # print("# ------------------- Data preprocessing ------------------- #")
    # print(f"rasterization: \033{colors[0]}{timeDict['rasterization']:.6f}\033[0m s, ")
    # total_time += timeDict['rasterization']
    # timeDict.pop('rasterization')
    if 'depth2point' in timeDict:
        print(f"Depth2point: \033{colors[0]}{timeDict['depth2point']:.6f}\033[0m s, ")
        total_time += timeDict['depth2point']
        timeDict.pop('depth2point')
        
    print("# ------------------- Our searching and sampling ------------------- #")
    for idx, (key, val) in enumerate(timeDict.items()):
        if key == 'feature_query':
            print("# ------------------- Rendering ------------------- #")

        print(f"[Step{idx}]: \033{colors[idx%len(colors)]}{key}: \t{val*1e3:.3f}\033[0m ms, ")     
        total_time += val
        render_time += val
    print(f"# --------- \033[31mtotal: {total_time:.4f}s\t render speed: {1/render_time:.2f}FPS \033[0m --------- #")

def AnalyzeTime(timeCompute):
    initTimeCompute = {}
    iterNum = len(timeCompute)
    for key in timeCompute[0].keys():
        initTimeCompute[key] = 0
    for timeDict in timeCompute:
        for key in timeDict.keys():
            initTimeCompute[key] += timeDict[key]
    for key in initTimeCompute.keys():
        initTimeCompute[key] /= iterNum
    printTime(initTimeCompute)
    

def copy_python_files(src: str, dest: str):
    for dirpath, dirnames, fnames in os.walk(src):
        src_files = [os.path.join(dirpath, f) for f in fnames if f.endswith(".py")]
        if src_files:
            dest_dir = os.path.join(dest, dirpath.strip(src).strip("/"))
            os.makedirs(dest_dir, exist_ok=True)
            os.system(f"cp {' '.join(src_files)} {dest_dir}")

class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])

def load_ckpt(model, model_state_dict, pretrained_ckpt, load_pointnerf_ckpt=False):
    if not load_pointnerf_ckpt:
        for k in pretrained_ckpt:
            model_state_dict[k] = pretrained_ckpt[k]
            print(k, "is loaded from pretrained weight")
        model.load_state_dict(model_state_dict)
        return model
    else:
        for k in pretrained_ckpt:
            if k.startswith('neural_points'):
                continue
            model_state_dict[k] = pretrained_ckpt[k]
            # print(k, "is loaded from pretrained weight")
        model_state_dict['neural_points.pc_xyz'] = pretrained_ckpt['neural_points.xyz'].squeeze(0)
        model_state_dict['neural_points.pc_rgb'] = pretrained_ckpt['neural_points.points_color'].squeeze(0)
        model_state_dict['neural_points.pc_dir'] = pretrained_ckpt['neural_points.points_dir'].squeeze(0)
        model_state_dict['neural_points.pc_conf'] = pretrained_ckpt['neural_points.points_conf'].squeeze(0)
        model_state_dict['neural_points.pc_feat'] = pretrained_ckpt['neural_points.points_embeding'].squeeze(0)
        model.load_state_dict(model_state_dict)
        return model


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
def render_bk(deltas, z_vals, sigmas, rgbs, white_bg=False):
    # Convert these values using volume rendering (Section 4)
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (bs, N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (bs, N_rays, N_samples_)
    # noise = torch.randn_like(sigmas) * noise_std
    # compute alpha by the formula (3)
    alphas = 1-torch.exp(-deltas.contiguous().view(sigmas.shape)*torch.relu(sigmas)) # (bs, N_rays, N_samples_)
    # alphas = 1-torch.exp(-torch.relu(sigmas)) # (bs, N_rays, N_samples_)
    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[..., :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
    weights = \
        alphas * torch.cumprod(alphas_shifted, -1)[..., :-1] # (bs, N_rays, N_samples_)
    weights_sum = weights.squeeze(-1).sum(-1) # (bs, N_rays), the accumulated opacity along the rays
                                    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    # compute final weighted outputs
    rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
    depth_final = torch.sum(weights*z_vals, -1, keepdim=True) # (N_rays)

    if white_bg:
        rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)
    # alphas[:, :, -1] = torch.zeros_like(alphas[:, :, -1])
    return rgb_final, depth_final, weights_sum, alphas

def clamp_t(t_val, vsize=0.004):
    t_dist = t_val[:, 1:] - t_val[:, :-1]
    t_dist = torch.cat([t_dist, torch.full_like(t_dist[:, :1], vsize)], dim=-1)
    mask = torch.logical_or(t_dist < 1e-8, t_dist > 2 * vsize)
    mask = mask.to(torch.float32)
    t_dist = t_dist * (1.0 - mask) + mask * vsize

    return t_dist

class Clocker(object):
    def __init__(self,):
        self.prev_time = time.time()
    def __call__(self, name):
        torch.cuda.synchronize()
        self.current_time = time.time()
        print(f'{name} : {(self.current_time - self.prev_time)*1e3:.2f} ms')
        self.prev_time = self.current_time
        
def move_data_to_device(batch, device):
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch