import re
import torch
import os, abc, cv2
import numpy as np
from typing import List
from tqdm import tqdm
import open3d as o3d

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def dict_2_torchdict(d: dict):
    for key, val in d.items():
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
        elif isinstance(val, dict):
            val = dict_2_torchdict(val)

        d[key] = val
    return d

def get_ray_intersections(ray1, ray2):
    """
    calculates points on ray1 and ray2 where both rays are closest to another
    :param ray1: torch Tensor [orgx, orgy, orgz, dirx, diry, dirz]
    :param ray2: torch Tensor [orgx, orgy, orgz, dirx, diry, dirz]
    :return:
    """

    B = (ray2[:3] - ray1[:3]).unsqueeze(1)
    A = torch.stack((ray1[3:], -ray2[3:]), dim=-1)

    t1t2 = torch.linalg.lstsq(A, B).solution
    t1t2 = t1t2.flatten()

    x1 = ray1[:3] + ray1[3:] * t1t2[0]
    x2 = ray2[:3] + ray2[3:] * t1t2[1]

    return x1, x2

def dataroot(dataset):
    if dataset == 'nerf_synthetic':
        return r'/home/jiahao/nerf/data/nerfs/nerf_sythetic/'
        # return r'F:/3D/NeRF/data/nerf_sythetic'
    elif dataset == 'dtu':
        return r'/home/jiahao/data/dtu/mvs_training/dtu'
    # elif dataset == 'dtu':
    #     return r'/home/jiahao/3DReconstruction/depth/TransMVSNet/outputs/dtu_testing'
    elif dataset == 'llff':
        return r'/home/jiahao/nerf/data/nerfs/llff/'
    
def triangulation_bpa(pnts, full_comb=True):
    # full_comb of Truck and TanksandTemple is 0(False), full_comb of nerf_synthetic_data is 1(True), full_comb of dtu is 2(True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pnts[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(pnts[:, :3] / np.linalg.norm(pnts[:, :3], axis=-1, keepdims=True))
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    radius = 3 * avg_dist
    dec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(
        [radius, radius * 2]))
    triangles = np.asarray(dec_mesh.triangles, dtype=np.int32)
    if full_comb:
        q, w, e = triangles[..., 0], triangles[..., 1], triangles[..., 2]
        triangles2 = np.stack([w,q,e], axis=-1)
        triangles3 = np.stack([e,q,w], axis=-1)
        triangles = np.concatenate([triangles, triangles2, triangles3], axis=0)
    return triangles
    

def search_nearest_poses(c2ws, num_src):
    def rotation_difference(R1, R2):
        R_diff = np.matmul(R1, np.transpose(R2, (0, 2, 1)))
        angle_diff = np.arccos((np.trace(R_diff, axis1=1, axis2=2) - 1) / 2) * (180/np.pi)
        return angle_diff
    
    def translation_difference(t1, t2):
        return np.linalg.norm(t1 - t2, axis=-1)
    nearest_poses = []
    Rs = c2ws[:, :3, :3]
    Ts = c2ws[:, :3, 3]
    for cam_idx in range(c2ws.shape[0]):
        t = Ts[cam_idx:cam_idx+1]
        dist_diff = translation_difference(t, Ts)
        sorted_indices_by_dist = np.argsort(dist_diff)
        nearest_indices_by_dist = sorted_indices_by_dist[1:num_src*2+1]
        
        R = Rs[cam_idx]
        angle_diff = rotation_difference(R, Rs[nearest_indices_by_dist])

        sorted_indices_by_angle = np.argsort(angle_diff)
        nearest_index = nearest_indices_by_dist[sorted_indices_by_angle[:num_src]]
        nearest_poses.append(np.array([cam_idx, *nearest_index]))
    
    return nearest_poses
