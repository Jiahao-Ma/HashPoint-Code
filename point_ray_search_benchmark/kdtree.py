import numpy as np
import matplotlib.pyplot as plt
import graphviz
import time
from tools import *
class Node:
    def __init__(self, points=None, left=None, right=None, isLeafNode=False, bbox=None, axis=None):
        self.points = points
        self.left = left
        self.right = right
        self.isLeafNode = isLeafNode
        self.bbox = bbox
        self.axis = axis
    

class KDTreePointSearch:
    def __init__(self, points, max_pc_per_node=10):
        points_ = [Point(i, point) for i, point in enumerate(points)]
        # self.root = self.build_kdtree(points_, max_points_per_node=10)
        self.max_pc_per_node = max_pc_per_node
        self.root = self.build_kdtree(points_)
    
    def build_kdtree(self, points, depth=0):
        n = len(points)
        if n <= 0:
            return None
        dimension = len(points[0].position)
        dim = depth % dimension
        if n <= self.max_pc_per_node:
            return Node(points=points, axis=dim)
        
        sorted_points = sorted(points, key=lambda point: point.position[dim])
        
        root = Node(points=sorted_points[n//2], axis=dim)
        root.left = self.build_kdtree(points = sorted_points[:n//2], depth=depth+1)
        root.right = self.build_kdtree(points = sorted_points[n//2+1:], depth=depth+1)
        return root

    
    def visualize_2d_kdtree(self, root:Node, min_x, max_x, min_y, max_y):
        lines = []
        def draw_all_kdtree(root:Node, min_x, max_x, min_y, max_y, lines, depth=0):
            if not root: 
                return
            dim = depth % 2
            if not isinstance(root.points, list):
                x, y = root.points.position
            else:
                sorted_points = sorted(root.points, key=lambda point: point.position[dim])
                median = sorted_points[len(sorted_points)//2].position
                x, y = median
            if root.axis == 0:
                # plt.plot([x, x], [min_y, max_y], 'black')
                lines.append([x, x, min_y, max_y])
                draw_all_kdtree(root.left, min_x, x, min_y, max_y, lines, depth+1)
                draw_all_kdtree(root.right, x, max_x, min_y, max_y, lines, depth+1)
            else:
                lines.append([min_x, max_x, y, y])
                # plt.plot([min_x, max_x], [y, y], 'black')
                draw_all_kdtree(root.left, min_x, max_x, min_y, y, lines, depth+1)
                draw_all_kdtree(root.right, min_x, max_x, y, max_y, lines, depth+1)
        draw_all_kdtree(root, min_x, max_x, min_y, max_y, lines, 0)
        return lines

    def k_nearest_search(self, point, k:int=1):
        neighbors = []

        def _search(root:Node, depth:int=0):
            if root is None:
                return

            dim = depth % len(point)
            axis_distance = root.point.position[dim] - point[dim]
            dist = np.linalg.norm(np.array(root.point.position) - np.array(point))

            if len(neighbors) < k:
                neighbors.append((dist, root.point))
                neighbors.sort(key=lambda x: x[0])
            elif dist < neighbors[-1][0]:
                neighbors[-1] = (dist, root.point)
                neighbors.sort(key=lambda x: x[0])

            if axis_distance < 0:
                _search(root.right, depth + 1)
                if abs(axis_distance) < neighbors[-1][0]:
                    _search(root.left, depth + 1)
            else:
                _search(root.left, depth + 1)
                if abs(axis_distance) < neighbors[-1][0]:
                    _search(root.right, depth + 1)

        _search(self.root)
        return neighbors
    
    def radius_search(self, point, radius:float):
        neighbors = []
        
        def _search(root:Node, depth:int=0, neighbors=None):
            if root is None:
                return
            
            for p in root.points:
                dist = np.linalg.norm(np.array(p.position) - np.array(point))

                if dist < radius:
                    neighbors.append((p.name))
                    # neighbors.add(p.name)
            _search(root.right, depth + 1, neighbors)
            _search(root.right, depth + 1, neighbors)
        
        _search(self.root, 0, neighbors)
        return set(neighbors)
    
    def ray_search_radius_vector3d(self, ray_o, ray_d, radius, t_near=2, t_far=6, num_sp=128):
       
        '''
            Uniformly sampling along the ray and search neighbors within the radius for each sampling point
        '''
        points = ray_o + ray_d * np.linspace(t_near, t_far, num_sp).reshape(-1, 1)
        output = []
        for point in points:
            output.extend(self.radius_search(point, radius))
        return set(output)
    
    def ray_search_radius_vector2d(self, ray_o, ray_d, radius, t_near=0, t_far=1000, num_sp=128):
        return self.ray_search_radius_vector3d(ray_o, ray_d, radius, t_near, t_far, num_sp)

class KDTreeRaySearch:
    '''
        This version is for ray search
    '''
    def __init__(self, points, max_points_per_node=300, max_depth=30):
        if points.shape[1] == 6:
            points_ = [Point(i, point[:3], point[3:]) for i, point in enumerate(points)]
        else:
            points_ = [Point(i, point) for i, point in enumerate(points)]
        
        self.max_points_per_node = max_points_per_node
        self.max_depth = max_depth 
        self.root = self.build_kdtree(points_, max_points_per_node)


    def build_kdtree(self, points, max_points_per_node, depth=0):
        # step1: construct bbox for points
        bbox = BBox(points)
        node = Node(bbox = bbox)

        # step2: construct root node
        if (len(points) <= self.max_points_per_node or depth >= self.max_depth):
            node.points = points
            node.isLeafNode = True
            return node
        
        # step3: split points into two parts
        dim = depth % len(points[0].position)
        sorted_points = sorted(points, key=lambda point: point.position[dim])
        middle = len(points) // 2

        node.left = self.build_kdtree(sorted_points[:middle], max_points_per_node, depth+1)
        node.right = self.build_kdtree(sorted_points[middle:], max_points_per_node, depth+1)
        return node

    def ray_search_radius_vector3d(self, node, ray_o, ray_d, radius):
        nearby_points = []
        def _search(node, ray_o, ray_d, radius, nearby_points):
            if not node:
                return 
            if node.isLeafNode or node.points is not None:
                    # search the point in current node
                    for point in node.points:
                        if dist_point2ray(point.position, ray_o, ray_d) <= radius:
                            nearby_points.append(point)
            # search left node
            if node.left is not None and AABB_ray(ray_o, ray_d, node.left.bbox.bb_min, node.left.bbox.bb_max, radius=radius):
                _search(node.left, ray_o, ray_d, radius, nearby_points)
            # search right node
            if node.right is not None and AABB_ray(ray_o, ray_d, node.right.bbox.bb_min, node.right.bbox.bb_max, radius=radius):
                _search(node.right, ray_o, ray_d, radius, nearby_points)
        
        _search(node, ray_o, ray_d, radius, nearby_points)
        return nearby_points

    def ray_search_radius_vector2d(self, node, ray_o, ray_d, radius):
        return self.ray_search_radius_vector3d(node, ray_o, ray_d, radius)

    def visualize_kdtree3d(self, node, geometries, vertexes):
        '''
            visualize kdtree
        '''
        if node is None:
            return
        else:
            if node.points is not None:
                points = o3d.geometry.PointCloud()
                points_xyz = np.array([point.position for point in node.points])
                points_rgb = np.array([point.color for point in node.points])
                points.points = o3d.utility.Vector3dVector(points_xyz)
                points.colors = o3d.utility.Vector3dVector(points_rgb)
                vertexes.append(points)
            aabb_min, aabb_max = node.bbox.bb_min, node.bbox.bb_max
            aabb = o3d.geometry.AxisAlignedBoundingBox(
                np.array(aabb_min), np.array(aabb_max))
            aabb.color = (0, 0, 0)
            geometries.append(aabb)
            self.visualize_kdtree3d(node.left, geometries, vertexes)
            self.visualize_kdtree3d(node.right, geometries, vertexes)
    
    def visualize_kdtree2d(self, node, geometries, vertexes, color=[0, 0, 0]):
        if node is None:
            return 
        else:
            if node.points is not None:
                points_xyz = np.array([point.position for point in node.points])
                vertexes.append(points_xyz)
            aabb_min, aabb_max = node.bbox.bb_min, node.bbox.bb_max
            # get 4 lines for aabb
            lines = []
            width = aabb_max[0] - aabb_min[0]
            '''
                aabb_min ------- *
                    |            |
                    |            |
                    |            |
                    * -----------aabb_max
            '''
            lines.append([[aabb_min, aabb_min + np.array([width, 0])],
                          [aabb_min + np.array([width, 0]), aabb_max],
                          [aabb_max, aabb_max - np.array([width, 0])],
                          [aabb_max - np.array([width, 0]), aabb_min]
                          ])
            # colors = [color for _ in range(4)]
            geometries.append(lines)
            self.visualize_kdtree2d(node.left, geometries, vertexes, color=color)
            self.visualize_kdtree2d(node.right, geometries, vertexes, color=color)
            
        
    
def demo1():
    # open3d implementation
    pc_path = r'pointnerf_ckpt/nerf_synthesis/tiny_lego.ply'
    pcd = o3d.io.read_point_cloud(pc_path)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    pcd_min = np.min(pcd.points, axis=0)
    pcd_max = np.max(pcd.points, axis=0)
    # randomly generate points within the range of pcd_min and pcd_max
    target = np.random.uniform(pcd_min, pcd_max, size=(1, 3)).astype(np.float32)
    t0 = time.time()
    [k, idx, _] = pcd_tree.search_knn_vector_3d(target.reshape(3, 1), 3)
    t1 = time.time()
    point_np = np.array(pcd.points)
    print(idx, point_np[idx])
    print('o3d time: ', t1 - t0)
    kdtree = KDTreePointSearch(point_np)
    # root = kdtree.build_kdtree(point_np)
    t2 = time.time()
    print(kdtree.k_nearest_search(target.reshape(-1), k=3))
    t3 = time.time()
    print('kdtree time: ', t3 - t2)

def demo2():
    # Create KDTree using your provided code...
    pc_path = r'pointnerf_ckpt/nerf_synthesis/lego.ply'
    pcd = o3d.io.read_point_cloud(pc_path)
    # randomly shuffle the points with index
    points_xyz = np.array(pcd.points)
    points_rgb = np.array(pcd.colors)
    points = np.concatenate([points_xyz, points_rgb], axis=1)
    np.random.shuffle(points)
    points = points[:10000]
    kdtree = KDTreeRaySearch(points, max_points_per_node=300)  
    
    voxels = []
    vertexes = []
    kdtree.visualize_kdtree3d(kdtree.root, voxels, vertexes)
    geometries = voxels + vertexes
    
    # visualize online
    o3d.visualization.draw_geometries(geometries)

def demo2_():
    # fix random points
    np.random.seed(0)
    points = np.random.rand(100, 2) * 100  

    kdtree = KDTreeRaySearch(points, max_points_per_node=30, max_depth=2)

    ray_origin = np.array([0, 75])
    ray_direction = np.array([1, -0.5])
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    thresh = 10.0
    nearby_points = kdtree.ray_search_radius_vector2d(kdtree.root, ray_origin, ray_direction, thresh)
    
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
    geometries = []
    vertexes = []
    kdtree.visualize_kdtree2d(kdtree.root, geometries, vertexes)
    geometries = np.array(geometries).reshape(-1, 4)
    color = 'black'
    for line in geometries:
        ax.plot([line[0], line[2]], [line[1], line[3]], color=color, linewidth=1)
    plt.gca().invert_yaxis()  # Coordinate system adjustment
    plt.axis('off')
    plt.show()

def demo3():
    np.random.seed(0)
    points = np.random.rand(100, 2) * 100  

    kdtree = KDTreePointSearch(points)

    ray_origin = np.array([0, 75])
    ray_direction = np.array([1, -0.5])
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    thresh = 30.0
    nearby_points = kdtree.ray_search_radius_vector2d(ray_origin, ray_direction, thresh)

    fig, ax = plt.subplots(figsize=(10,10))
    kdtree.visualize_2d_kdtree(kdtree.root, min_x = 0, max_x = 100, min_y = 0, max_y = 100)

    # visualize boundary
    ax.plot([0, 100], [0, 0], color='black', linewidth=1)
    ax.plot([0, 0], [0, 100], color='black', linewidth=1)
    ax.plot([0, 100], [100, 100], color='black', linewidth=1)
    ax.plot([100, 100], [0, 100], color='black', linewidth=1)
    try:
        nearby_point_indices = [point.name for point in nearby_points]
    except:
        nearby_point_indices = nearby_points
    # visualize all points
    for idx, point in enumerate(points):
        if not idx in nearby_point_indices:
            ax.scatter(point[0], point[1], facecolors='none', edgecolors='black', s=50, linewidths=1.5)
        else:
            # visualize nearby points
            ax.scatter(point[0], point[1], color='red', s=50, cmap='hsv', alpha=0.6)



    plt.show()
if __name__ == '__main__':
    import open3d as o3d
    import numpy as np
    
    # demo1()

    # demo2()
    
    # demo2_()

    demo3()
