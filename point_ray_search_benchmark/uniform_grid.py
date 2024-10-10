import sys, os; sys.path.append(os.getcwd()); os.environ['CUDA_VISIBLE_DEVICES'] = '1'
try:
    sys.path.remove('/home/jiahao/nerf/code/VGNF/epcq/epcq_grid')
except:
    pass

from tools import *

class UniformGrid:
    def __init__(self, points, voxel_size):
        self.voxel_size = voxel_size
        self.min_bound = np.min(points, axis=0)
        self.max_bound = np.max(points, axis=0)
        self.grid_size = np.ceil((self.max_bound - self.min_bound) / voxel_size).astype(int)
        self.grid = [[] for _ in range(np.prod(self.grid_size))]
        self.points = points
        # build uniform grid
        for i, point in enumerate(points):
            voxel_index = self.get_voxel_index(point)
            self.grid[voxel_index].append(Point(i, point))
        # print(' complete building uniform grid !')
        # print(f' {len(self.grid)} voxel grids in total. grid size: {self.grid_size}')
        # for idx, vxl in enumerate(self.grid):
        #     print(f'[{idx}] voxel grid has {len(vxl)} points.')

    def get_voxel_index(self, point):
        voxel_index = ((point - self.min_bound) / self.voxel_size).astype(int)
        voxel_index = np.minimum(voxel_index, self.grid_size - 1)
        return np.ravel_multi_index(voxel_index, self.grid_size)
    
    def get_voxel_index_ijk(self, voxel_index):
        return np.unravel_index(voxel_index, self.grid_size)

    def find_start_cell(self, ray_o, ray_d):
        t_values_min = [(self.min_bound[i] - ray_o[i]) / ray_d[i] for i in range(3)]
        t_values_max = [(self.max_bound[i] - ray_o[i]) / ray_d[i] for i in range(3)]
        t_entry = max(min(t_values_min[i], t_values_max[i]) for i in range(3))
        entry_point = [ray_o[i] + t_entry * ray_d[i] for i in range(3)]
        return [int((entry_point[i] - self.min_bound[i]) / self.voxel_size) for i in range(3)]

    def ray_search_radius_vector3d(self, ray_origin, ray_direction, max_distance):
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        nearby_poitns = []
        # 3D DDA
        step = [1 if ray_direction[i] > 0 else -1 for i in range(3)]
        t_max = [(self.min_bound[i] + (1 if step[i] > 0 else 0) ** self.grid_size[i] - ray_origin[i]) / ray_direction[i] for i in range(3)]
        t_delta = [self.voxel_size / abs(ray_direction[i]) for i in range(3)]
        cell = self.find_start_cell(ray_origin, ray_direction)
        while 0 <= cell[0] < self.grid_size[0] and 0 <= cell[1] < self.grid_size[1] and 0 <= cell[2] < self.grid_size[2]:
            axis = t_max.index(min(t_max))
            t_max[axis] += t_delta[axis]
            voxel_index = np.ravel_multi_index(cell, self.grid_size)
            points = self.grid[voxel_index]
            for point in points:
                if (dist_point2ray(point.position, ray_origin, ray_direction) <= max_distance):
                    nearby_poitns.append(point)
            cell[axis] += step[axis]
        # # Iterate through each grid cell
        # for i in range(self.grid_size[0]):
        #     for j in range(self.grid_size[1]):
        #         for k in range(self.grid_size[2]):
        #             # Compute the min and max bounds of this voxel
        #             aabb_min = self.min_bound + np.array([i, j, k]) * self.voxel_size
        #             aabb_max = aabb_min + self.voxel_size
        #             # Check for ray intersection
        #             if AABB_ray(ray_origin, ray_direction, aabb_min, aabb_max):
        #                 # Convert i, j, k to a single index
        #                 voxel_index = np.ravel_multi_index([i, j, k], self.grid_size)
        #                 # Add the points in this voxel to the output list
        #                 points = self.grid[voxel_index]
        #                 for point in points:
        #                     if (dist_point2ray(point.position, ray_origin, ray_direction) <= max_distance):
        #                         nearby_poitns.append(point)
                        
        return nearby_poitns
    
    def ray_search_radius_vector2d(self, ray_o, ray_d, thresh):
        ray_d = ray_d / np.linalg.norm(ray_d)
        nearby_poitns = []
        # Iterate through each grid cell
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                # Compute the min and max bounds of this voxel
                aabb_min = self.min_bound + np.array([i, j]) * self.voxel_size
                aabb_max = aabb_min + self.voxel_size
                # Check for ray intersection
                if AABB_ray(ray_o, ray_d, aabb_min, aabb_max):
                    # Convert i, j, k to a single index
                    voxel_index = np.ravel_multi_index([i, j], self.grid_size)
                    # Add the points in this voxel to the output list
                    points = self.grid[voxel_index]
                    for point in points:
                        if (dist_point2ray(point.position, ray_o, ray_d) <= thresh):
                            nearby_poitns.append(point)
        return nearby_poitns
    '''
        3D visualization
    '''
    def draw_single_voxel(self, coord, voxel_size, color=[0,0,0]):
        lines = []
        corners = [
            coord,
            coord + [voxel_size, 0, 0],
            coord + [0, voxel_size, 0],
            coord + [0, 0, voxel_size],
            coord + [voxel_size, voxel_size, 0],
            coord + [voxel_size, 0, voxel_size],
            coord + [0, voxel_size, voxel_size],
            coord + [voxel_size, voxel_size, voxel_size]
        ]

        # Bottom face
        lines.extend([corners[0], corners[1]])
        lines.extend([corners[1], corners[4]])
        lines.extend([corners[4], corners[2]])
        lines.extend([corners[2], corners[0]])

        # Top face
        lines.extend([corners[3], corners[5]])
        lines.extend([corners[5], corners[7]])
        lines.extend([corners[7], corners[6]])
        lines.extend([corners[6], corners[3]])

        # Vertical lines
        lines.extend([corners[0], corners[3]])
        lines.extend([corners[1], corners[5]])
        lines.extend([corners[2], corners[6]])
        lines.extend([corners[4], corners[7]])

        colors = [color for _ in range(12)]

        return lines, colors

    def draw_all_voxel_grid(self, color=[0,0,0]):
        lines = []
        colors = []
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                for k in range(self.grid_size[2]):
                    coord = self.min_bound + np.array([i, j, k]) * self.voxel_size
                    ls, cs = self.draw_single_voxel(coord, self.voxel_size, color)
                    lines.extend(ls)
                    colors.extend(cs)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack(lines))
        line_set.lines = o3d.utility.Vector2iVector(np.arange(len(lines)).reshape(-1, 2))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set
    
    def draw_single_voxel_grid(self, voxel_indices, color=[1, 0, 0]):
        lines = []
        colors = []
        for vxl_idx in voxel_indices:
            vxl_idx_ijk = self.get_voxel_index_ijk(vxl_idx)
            coord = self.min_bound + np.array(vxl_idx_ijk) * self.voxel_size
            ls, cs = self.draw_single_voxel(coord, self.voxel_size, color)
            lines.extend(ls)
            colors.extend(cs)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack(lines))
        line_set.lines = o3d.utility.Vector2iVector(np.arange(len(lines)).reshape(-1, 2))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set
    
    def draw_ray(self, ray_o, ray_d, t_near, t_far, color=[0, 1, 0]):
        line_set = o3d.geometry.LineSet()
        ray_start = ray_o + ray_d * t_near
        ray_end = ray_o + ray_d * t_far
        line = np.array([ray_start, ray_end])
        line_set.points = o3d.utility.Vector3dVector(line)
        line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_set.colors = o3d.utility.Vector3dVector([color])

        return line_set

    '''
        2D visualization
    '''
    def draw_single_grid(self, coord, voxel_size, color=[0,0,0]):
        lines = []
        corners = [
            coord,
            coord + [voxel_size[0], 0],
            coord + [0, voxel_size[1]],
            coord + [voxel_size[0], voxel_size[1]]
        ]

        # Bottom face
        lines.extend([corners[0], corners[1]])
        lines.extend([corners[1], corners[3]])
        lines.extend([corners[3], corners[2]])
        lines.extend([corners[2], corners[0]])

        colors = [color for _ in range(4)]

        return lines, colors
    
    def draw_all_grids(self, color=[0,0,0]):
        lines = []
        colors = []
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                coord = self.min_bound + np.array([i, j]) * self.voxel_size
                ls, cs = self.draw_single_grid(coord, self.voxel_size, color)
                lines.extend(ls)
                colors.extend(cs)


        return lines, colors
    
def demo1():
    args = parser.parse_args()
    
    img_index = 0
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
    voxel_size = 0.2
    ug = UniformGrid(pcd, voxel_size)
    t1 = time.time()
    print(f'build uniform grid: {t1 - t0:.6f}s')

    '''
        Searching for points near a ray
    '''
    threshold = 0.016
    # define the pixel index
    # x, y = np.meshgrid(np.arange(cam.width), np.arange(cam.height), indexing='xy')
    # xy = np.stack([x, y], axis=-1).reshape(-1, 2)
    # np.random.shuffle(xy)
    # # randomly select n groups of pixel index from the image
    # n = 100
    # xy = [[400, 400]]
    # for uv in tqdm(xy[:n]):

    uv = [400, 400]
    ray_o = rays_o[uv[1], uv[0]].cpu().numpy()
    ray_d = rays_d[uv[1], uv[0]].cpu().numpy()
    t2 = time.time()
    voxel_indices, nearby_poitns = ug.ray_search_radius_vector3d(ray_o, ray_d, threshold)
    t3 = time.time()
    # print(f'Found {len(voxel_indices)} intersected voxels including: {voxel_indices}')
    # print(len(nearby_poitns))
    t4 = time.time()
    correct_nearby_points = correct_answer(pcd, ray_o, ray_d, threshold)
    t5 = time.time()
    print(f'violent_enumeration searching results: {len(correct_nearby_points)} points\nUniform grid searching results: {len(nearby_poitns)} points')
    print(f'AABB + uniform grid: {t3 - t2:.6f}s', )
    print(f'violent enumeration: {t5 - t4:.6f}s', )

    '''
        Visualize the voxel grid
    '''
    vis_list = []

    # visualize all voxels
    # all_voxel_lineset = ug.draw_all_voxel_grid()
    # vis_list.append(all_voxel_lineset)
    
    # visualize intersected voxels
    intersect_voxel_lineset = ug.draw_single_voxel_grid(voxel_indices)
    vis_list.append(intersect_voxel_lineset)
    
    # visualize rays
    line = ug.draw_ray(ray_o, ray_d, 2.0, 6.0)
    vis_list.append(line)

    # Add points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd[:1000])
    vis_list.append(point_cloud)

    # Show everything
    
    o3d.visualization.draw_geometries(vis_list)

def demo2():
    def convert_2d_to_3d(points_2d):
        return [(x, y, 0) for x, y in points_2d]
    '''
        Generate 2D points on the plane and use uniform grid to search
    '''
    points = np.random.rand(200, 2) * 1000  

    voxel_size = np.array([100, 100])
    uniform_grid = UniformGrid(points, voxel_size)

    ray_origin = np.array([0, 0])
    ray_direction = np.array([1, 1])
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    thresh = 20.0
    nearby_points = uniform_grid.ray_search_radius_vector2d(ray_origin, ray_direction, thresh)
    
    print(f"Total nearby points: {len(nearby_points)}")

    # Visualization using OpenCV
    img_size = 1010  # 设定图像尺寸
    img = np.ones((img_size, img_size, 3), np.uint8) * 255  # 创建白色背景

    # 绘制每个点
    thickness = 2  
    radius = 3  
    for point in points:
        x, y = point.astype(int)
        cv2.circle(img, (x, y), radius, (0, 0, 0), thickness)  # 黑色表示其他点的边

    thickness = 4  
    radius = 3  
    for point in nearby_points:
        x, y = point.position.astype(int)
        cv2.circle(img, (x, y), radius, (0, 0, 255), thickness)

    lines, colors = uniform_grid.draw_all_grids()
    lines = np.array(lines).reshape(-1, 4)
    for line, color in zip(lines, colors):
        cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])),  color, 1)
    cv2.imshow("img", img)
    cv2.waitKey(0)

def demo2_():
    '''
        Generate 2D points on the plane and use uniform grid to search
    '''
    points = np.random.rand(100, 2) * 100  

    voxel_size = np.array([25, 25])
    uniform_grid = UniformGrid(points, voxel_size)

    ray_origin = np.array([0, 75])
    ray_direction = np.array([1, -0.5])
    # ray_direction = ray_direction / np.linalg.norm(ray_direction)

    thresh = 10.0
    nearby_points = uniform_grid.ray_search_radius_vector2d(ray_origin, ray_direction, thresh)
    
    print(f"Total nearby points: {len(nearby_points)}")

    fig, ax = plt.subplots(figsize=(10,10))
    
    nearby_point_indices = [point.name for point in nearby_points]
    
    # visualize all points 
    for idx, point in enumerate(points):
        if not idx in nearby_point_indices:
            ax.scatter(point[0], point[1], facecolors='none', edgecolors='black', s=50, linewidths=1.5)

    # visualize nearby points
    for point in nearby_points:
        ax.scatter(point.position[0], point.position[1], color='red', s=50, cmap='hsv', alpha=0.6)

    # visualize grid
    lines, colors = uniform_grid.draw_all_grids()
    lines = np.array(lines).reshape(-1, 4)
    for line, color in zip(lines, colors):
        ax.plot([line[0], line[2]], [line[1], line[3]], color=color, linewidth=1)

    # visualize ray
    # ray_end = ray_origin + 100 * ray_direction
    # ray_start = ray_origin 
    # ax.plot([ray_start[0], ray_start[1]], [ray_end[0], ray_end[1]], color='blue', linewidth=1)

    plt.gca().invert_yaxis()  # Coordinate system adjustment
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    demo2_()
    