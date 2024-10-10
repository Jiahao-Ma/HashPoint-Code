import os, json
import cv2
import torch
import time
import random
import numpy as np
import open3d as o3d
from tqdm import tqdm
from dataset.nerfsynthesis import NeRFSynthesis
from epcq.epcq import CameraSpec
from utils.config import parser
import torch
import numpy as np
import matplotlib.pyplot as plt

class Point(object):
    def __init__(self, name, position, color=None) -> None:
        self.name = name
        self.position = position
        self.color = color
    def __str__(self):
        return "name: {} position: {}".format(self.name, self.position)

class BBox(object):
    def __init__ (self, points) -> None:
        # calculate the bounding box of the points
        # points: list of Point
        self.bb_min = np.min([point.position for point in points], axis=0)
        self.bb_max = np.max([point.position for point in points], axis=0)
        
        
    
def generate_rays(cam:CameraSpec):
    x,y = torch.meshgrid([torch.arange(cam.width, dtype=torch.float32, device=cam.fx.device),
                            torch.arange(cam.height, dtype=torch.float32, device=cam.fx.device)],
                            indexing='xy',
                            )
    rays = torch.stack([ (x - cam.cx) / cam.fx, (y - cam.cy) / cam.fy, torch.ones_like(x)], dim=-1)
    rays_d = torch.sum(rays[:, :, None, :] * cam.c2w[:3, :3][None, None, :, :], dim=-1)
    rays_o = torch.Tensor(cam.c2w[:3, 3]).expand_as(rays_d)
    return rays_o, rays_d

def AABB_ray(ray_origin, ray_direction, aabb_min, aabb_max, epsilon=1e-8, radius=None):
    '''
        AABB algorithm:
            judge if the ray have the intersection with the axis-aligned bounding box
    '''
    if radius is not None:
        tmin = (aabb_min - radius - ray_origin) / (ray_direction + epsilon)
        tmax = (aabb_max + radius- ray_origin) / (ray_direction + epsilon)
    else:
        tmin = (aabb_min - ray_origin) / (ray_direction + epsilon)
        tmax = (aabb_max - ray_origin) / (ray_direction + epsilon)
    
    t_enter = np.minimum(tmin, tmax)
    t_exit  = np.maximum(tmin, tmax)
    
    t_enter = np.maximum.reduce(t_enter)
    t_exit  = np.minimum.reduce(t_exit)
    
    return t_enter < t_exit and t_exit >= 0

def dist_point2ray(point, ray_origin, ray_direction):
    '''
        calculate the distance between ray and point
    '''
    v = point - ray_origin
    ray_direction = ray_direction / np.linalg.norm(ray_direction, axis=-1)
    P_proj = ray_origin + np.dot(v, ray_direction) * ray_direction
    distance = np.linalg.norm(point - P_proj)
    return distance

def correct_answer(points, ray_o, ray_d, threshold):
    nearby_points = []
    for point in points:
        if (dist_point2ray(point, ray_o, ray_d) <= threshold):
            nearby_points.append(point)
    return nearby_points

def violent_enumeration(target, points):
    dist = np.linalg.norm(target - points, axis=-1)
    min_dist_index = np.argmin(dist)
    nearest_point = points[min_dist_index]
    nearest_dist = dist[min_dist_index]
    
    return nearest_dist, nearest_point

def violent_enumeration_ray(ray_o, ray_d, points, radius):
    points_c = [Point(i, p) for i, p in enumerate(points)]
    sub_p_o = points - ray_o.reshape(1, -1)
    ray_d = (ray_d / np.linalg.norm(ray_d)).reshape(1, -1)
    p_proj = ray_o + np.sum(sub_p_o * ray_d, axis=-1, keepdims=True) * ray_d
    dist = np.linalg.norm(p_proj - points, axis=-1)
    mask = dist <= radius
    nearest_points = points[mask]
    
    return nearest_points, np.array(points_c)[mask]