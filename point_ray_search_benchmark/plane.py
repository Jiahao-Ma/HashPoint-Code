import sys
try:
    sys.path.remove('/home/jiahao/nerf/code/VGNF/epcq/epcq_grid')
except:
    pass
import numpy as np
from tools import *
from pytorch_lightning.utilities import move_data_to_device
from epcq.epcq import _C, CameraSpec, RenderOptions, utils, NeuralPoints


class Plane(object):
    def __init__(self, K, w2c, height, width, fx, fy) -> None:
        self.K = K
        self.fx = K[0,0]
        self.fy = K[1,1]
        self.cx = K[0,2]
        self.cy = K[1,2]
        if w2c.shape == (3, 4):
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0)
        self.c2w = np.linalg.inv(w2c)
        self.w2c = w2c
        self.height = height
        self.width = width
        self.fx = fx
        self.fy = fy
        self.plane = [[] for _ in range(height * width)]

    def index2uv(self, index):
        u = index % self.width
        v = index // self.width
        return u, v
    
    def uv2index(self, u, v):
        return v * self.width + u

    def insertNode(self, index, point):
        self.plane[index].append(point)

    def calculate_pixel_length(self, ):
        x, y = np.meshgrid(np.arange(self.width, dtype=np.float32), np.arange(self.height, dtype=np.float32), indexing='xy')
        camera_dirs = np.stack([(x - self.cx) / self.fx, (y - self.cy) / self.fy, np.ones_like(x)], axis=-1)
        directions = camera_dirs @ self.c2w[:3, :3].T
        dx = np.sqrt(np.sum((directions[:-1, :, :] - directions[1:, :, :])**2, -1))
        dx = np.concatenate([dx, dx[-2:-1, :]], 0)
        return dx
    
class PlaneSearch(object):
    def __init__(self, points, w2c, K, height, width, fx, fy) -> None:
        self.plane = Plane(K, w2c, height, width, fx, fy)
        # points init
        plane_indices = self.world2plane(points, w2c, K)
        for point_idx, (plane_idx, point) in enumerate(zip(plane_indices, points)):
            self.plane.insertNode(plane_idx, Point(point_idx, point))
        # calculate the length of each pixel
        # self.pixel_lens = self.plane.calculate_pixel_length()

    def world2plane(self, points, w2c, K):
        '''
            Args:   
                points: (n, 3)
                w2c: (4, 4)
                K: (3, 3)
        '''
        # (3, 3) @ (3, n) + (3, 1) = (3, n)
        points_c = w2c[:3, :3] @ points.T + w2c[:3, 3:4]
        # (3, 3) @ (3, n) = (3, n)
        points_s = K @ points_c
        # (3, n) -> (2, n)
        points_s = points_s / points_s[2:3, :]
        u_index = points_s[0, :].astype(np.int32)
        v_index = points_s[1, :].astype(np.int32)
        index = self.plane.uv2index(u_index, v_index)
        return index

    def dist_point2ray(self, pt, ray_o, ray_d):
        ray_d = ray_d / np.linalg.norm(ray_d) # normalize the ray direction to unit vector
        sub_pt = pt - ray_o
        t = np.dot(sub_pt, ray_d)
        proj_pt = ray_o + t * ray_d
        return np.linalg.norm(proj_pt - pt)
    
    def ray_search_radius_vector3d(self, u_idx, v_idx, ray_o, ray_d, radius):
        plane_idx = self.plane.uv2index(u_idx, v_idx)
        # pixel_len = self.pixel_lens[v_idx, u_idx]
        # kernel_size = radius / pixel_len
        kernel_size = 12
        u_start = max(0, u_idx - kernel_size // 2)
        u_end = min(self.plane.width, u_idx + kernel_size // 2 + 1)
        v_start = max(0, v_idx - kernel_size // 2)
        v_end = min(self.plane.height, v_idx + kernel_size // 2 + 1)
        nearby_points = []
        for u_offset in range(u_start, u_end):
            for v_offset in range(v_start, v_end):
                plane_idx = self.plane.uv2index(u_offset, v_offset)
                for point in self.plane.plane[plane_idx]:
                    if self.dist_point2ray(point.position, ray_o, ray_d) <= radius:
                        nearby_points.append(point)
        return nearby_points
    
class PlaneSearchGPU(object):
    def __init__(self):
        self.rasterize_hashtable = _C.__dict__['scatter_hashtable']
        self.query_hashtable = _C.__dict__['quick_query_for_nearby_point']
    
    def _world2cam(self, cam:CameraSpec, pc_xyz:torch.Tensor=None):
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
    
    def build_data_structure(self, points:torch.Tensor, cam:CameraSpec, sort=False):
        pts_c, depth = self._world2cam(cam, points)
        _, _, screen_x_d, screen_y_d = self._cam2image(pts_c, cam)
        point2img_idx = self._convetpixel(screen_x_d, screen_y_d, [cam.height, cam.width])
        point_idx = torch.arange(points.shape[0], dtype=torch.int32, device=points.device)
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
    
    def query(self, points, cam:CameraSpec, options:RenderOptions, 
              sorted_point_idx_list:torch.Tensor, 
              hashtable:torch.Tensor, 
              indices_of_ray:torch.Tensor):
        if len(indices_of_ray.shape) == 2 and indices_of_ray.shape[1] == 2:
            indices_of_ray = indices_of_ray[:, 0] + indices_of_ray[:, 1] * cam.width
        elif indices_of_ray.shape == (2,):
            indices_of_ray = indices_of_ray[0] + indices_of_ray[1] * cam.width
        elif len(indices_of_ray.shape) == 1:
            pass
        else:
            raise ValueError("indices_of_ray should be 1D or 2D with shape (N, 2)")
        indices_of_ray = indices_of_ray.to(dtype=torch.int32).reshape(-1)
        
        if not isinstance(points, NeuralPoints):
            points = NeuralPoints(pc_type='pc', points = points)
        
        # sorted_point_idx_list_ = {'sorted_point_idx_list' : sorted_point_idx_list}
        # sorted_point_idx_list_ctn = torch.jit.script(utils.Container(sorted_point_idx_list_))
        # sorted_point_idx_list_ctn.save(r'/home/jiahao/nerf/code/VGNF/epcq/cuda_debug_epcq_grid1/variables1/sorted_point_idx_list.pt')
        
        # hashtable_ = {'hashtable' : hashtable}
        # hashtable_ctn = torch.jit.script(utils.Container(hashtable_))
        # hashtable_ctn.save(r'/home/jiahao/nerf/code/VGNF/epcq/cuda_debug_epcq_grid1/variables1/hashtable.pt')
        
        # indices_of_ray_ = {'indices_of_ray' : indices_of_ray}
        # indices_of_ray_ctn = torch.jit.script(utils.Container(indices_of_ray_))
        # indices_of_ray_ctn.save(r'/home/jiahao/nerf/code/VGNF/epcq/cuda_debug_epcq_grid1/variables1/indices_of_ray.pt')
        
        # points._save2torch(r"/home/jiahao/nerf/code/VGNF/epcq/cuda_debug_epcq_grid1/variables1/epcq.pt")
        # cam._save2torch(r'/home/jiahao/nerf/code/VGNF/epcq/cuda_debug_epcq_grid1/variables1/cam.pt')
        # options._save2torch(r'/home/jiahao/nerf/code/VGNF/epcq/cuda_debug_epcq_grid1/variables1/opt.pt')
        
        output_pc_idx = self.query_hashtable(points._to_cpp(), cam._to_cpp(),
                             options._to_cpp(),
                             sorted_point_idx_list,
                             hashtable,
                             indices_of_ray
                             )
        torch.cuda.synchronize()
        return output_pc_idx
    
def testRaySearching():
    args = parser.parse_args()
    status = 'inference'#'train_on_pc'
    nerfsynthesis = NeRFSynthesis(args, 'lego', 
                                  ref_downsample=1,
                                  src_downsample=1,
                                  split='train',
                                  status=status)
    data = nerfsynthesis[0]
    fx, fy, cx, cy = CameraSpec.decomposeK(data['ref_K'])
    cam = CameraSpec(c2w=data['ref_c2w'],
                        w2c=data['ref_w2c'],
                        K = data['ref_K'],
                        fx=fx, fy=fy,
                        cx=cx, cy=cy,
                        height=data['ref_hw'][0],
                        width=data['ref_hw'][1])
    rays_o, rays_d = generate_rays(cam)
    
    pc_path = r'pointnerf_ckpt/nerf_synthesis/tiny_lego.ply'
    pcd = o3d.io.read_point_cloud(pc_path)
    pcd = np.array(pcd.points)
    print("number of points: ", pcd.shape[0])
    t0 = time.time()
    epcq = PlaneSearch(points=pcd, 
                       w2c=cam.w2c.cpu().numpy(), 
                       K=cam.K.cpu().numpy(), 
                       height=cam.height.cpu().numpy(), 
                       width=cam.width.cpu().numpy(),
                       fx=cam.fx.cpu().numpy(),
                       fy=cam.fy.cpu().numpy())
    t1 = time.time()
    print(f"build epcq: {t1 - t0: .6f}s")
    threshold = 0.016
    uv = [400, 400]
    ray_o = rays_o[uv[1], uv[0]].cpu().numpy()
    ray_d = rays_d[uv[1], uv[0]].cpu().numpy()
    t2 = time.time()
    nearby_poitns = epcq.ray_search_radius_vector3d(uv[0], uv[1], ray_o, ray_d,  threshold)
    t3 = time.time()
    print(f"search nearby points: {t3 - t2: .6f}s")
    print(f"number of nearby points: {len(nearby_poitns)}")

def testRaySearchingGPU():
    args = parser.parse_args()
    status = 'inference'#'train_on_pc'
    nerfsynthesis = NeRFSynthesis(args, 'lego', 
                                  ref_downsample=1,
                                  src_downsample=1,
                                  split='train',
                                  status=status)
    data = move_data_to_device(nerfsynthesis[0], torch.device(args.device))
    fx, fy, cx, cy = CameraSpec.decomposeK(data['ref_K'])
    options = RenderOptions()
    cam = CameraSpec(c2w=data['ref_c2w'],
                        w2c=data['ref_w2c'],
                        K = data['ref_K'],
                        fx=fx, fy=fy,
                        cx=cx, cy=cy,
                        height=data['ref_hw'][0],
                        width=data['ref_hw'][1])
    rays_o, rays_d = generate_rays(cam)
    
    pc_path = r'pointnerf_ckpt/nerf_synthesis/tiny_lego.ply'
    pcd = o3d.io.read_point_cloud(pc_path)
    pcd = torch.from_numpy(np.array(pcd.points, dtype=np.float32)).cuda()
    uv = torch.Tensor([400, 400]).cuda()
    plane = PlaneSearchGPU()
    sorted_point_idx_list, sorted_depth_list, hash_table = plane.build_data_structure(pcd, cam, sort=False)
    output_pc_idx = plane.query(pcd, cam, options, sorted_point_idx_list, hash_table, uv)
    
    
if __name__ == '__main__':
    # testRaySearching()
    testRaySearchingGPU()

