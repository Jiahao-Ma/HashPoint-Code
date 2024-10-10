import numpy as np
from tools import *


class OctNode(object):
    def __init__(self, position, size, depth, data):
        '''
        OctNode Cubes
        
        branch: 0 1 2 3 4 5 6 7
        x:      - - - - + + + +
        y:      - - + + - - + +
        z:      - + - + - + - +
        '''
        self.position = position
        self.size = size
        self.depth = depth
        
        self.isLeafNode = True
        
        self.data = data
        
        self.branches = [None] * 8
        
        half = size / 2.0
        
        self.lower = np.array((position[0] - half, position[1] - half, position[2] - half))
        self.upper = np.array((position[0] + half, position[1] + half, position[2] + half))
        
    def __str__(self):
        data_str = u", ".join((str(d) for d in self.data))
        return u"OctNode(position={0}, size={1}, depth={2}, data=[{3}])"\
            .format(self.position, self.size, self.depth, data_str)

class Octree(object):
    def __init__(self, worldsize=None, origin=(0, 0, 0), leaf_data_size=10, points=None):
        if points is None:
            self.root = OctNode(origin, worldsize, 0, [])
            self.worldsize = worldsize
            self.limit = leaf_data_size
        else:
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            worldsize = max_bound - min_bound
            self.root = OctNode(min_bound + worldsize / 2, np.max(worldsize), 0, [])
            self.worldsize = np.max(worldsize)
            self.limit = leaf_data_size
            for point_idx, point in enumerate(points):
                if isinstance(point, np.ndarray):
                    point = Point(point_idx, point)

                self.insertNode(point.position, point)
        
    def insertNode(self, position, objData=None):
        if np.any(position < self.root.lower) or np.any(position > self.root.upper):
            return False
        if objData is None:
            objData = position
        
        return self._insertNode(self.root, self.root.size, self.root, position, objData)
    
    def _insertNode(self, root:OctNode, size:int, parent:OctNode, position:np.array, objData:np.array):
        if root is None:
            pos = parent.position
            
            offset = size / 2
            
            branch = self._findBranch(parent, position)
            
            newCenter = (0, 0, 0)
            
            if branch == 0:
                newCenter = (pos[0] - offset, pos[1] - offset, pos[2] - offset )
            elif branch == 1:
                newCenter = (pos[0] - offset, pos[1] - offset, pos[2] + offset )
            elif branch == 2:
                newCenter = (pos[0] - offset, pos[1] + offset, pos[2] - offset )
            elif branch == 3:
                newCenter = (pos[0] - offset, pos[1] + offset, pos[2] + offset )
            elif branch == 4:
                newCenter = (pos[0] + offset, pos[1] - offset, pos[2] - offset )
            elif branch == 5:
                newCenter = (pos[0] + offset, pos[1] - offset, pos[2] + offset )
            elif branch == 6:
                newCenter = (pos[0] + offset, pos[1] + offset, pos[2] - offset )
            elif branch == 7:
                newCenter = (pos[0] + offset, pos[1] + offset, pos[2] + offset )
            
            return OctNode(newCenter, size, parent.depth + 1, [objData])
            
        elif ( not root.isLeafNode and (root.position != position).all()):
            branch = self._findBranch(root, position)
            newSize = root.size / 2
            root.branches[branch] = self._insertNode(root.branches[branch], newSize, root, position, objData)
        
        elif root.isLeafNode:
            if (len(root.data) <self.limit):
                root.data.append(objData)
            
            else:
                root.data.append(objData)
                objList = root.data
                root.data = None
                root.isLeafNode = False
                newSize = root.size / 2
                for obj in objList:
                    branch = self._findBranch(root, obj.position)
                    root.branches[branch] = self._insertNode(root.branches[branch], newSize, root, obj.position, obj)
        
        return root
    
    def pt_search_radius_vector3d(self, node, position, radius):
        '''
            Search the nearby points in the octree within the radius
        '''
        def _pt_search_radius_vector3d(self, node, position, radius):
            nearby_pts = []
            if node.isLeafNode and node.data:
                for obj in node.data:
                    if np.linalg.norm(np.array(obj.position) - np.array(position)) <= radius:
                        nearby_pts.append(obj)
            else:
                for branch in node.branches:
                    if branch is not None and self._intersect(branch, position, radius):
                        nearby_pts.extend(_pt_search_radius_vector3d(self, branch, position, radius))
            
            return nearby_pts
        
        
        if np.any(position < self.root.lower) or np.any(position > self.root.upper):
            return []
        
        return _pt_search_radius_vector3d(self, node, position, radius)
    
    def ray_search_radius_vector3d(self, ray_o, ray_d, radius):
        '''
            Search the nearby point in the octree within the radius for ray tracing
            Steps:
                1. Search intersection between ray and voxel(cube) by AABB 
                2. Search intersection between ray and ball by sphere
        '''
        def _ray_search_radius_vector3d(self, node, ray_o, ray_d, radius):
            nearby_pts = []
            if node.isLeafNode and node.data:
                for obj in node.data:
                    if self.dist_point2ray(obj.position, ray_o, ray_d) <= radius:
                        nearby_pts.append(obj)
            else:
                for branch in node.branches:
                    if branch is not None and self._intersectAABB_withRadius(branch, ray_o, ray_d, radius):
                        nearby_pts.extend(_ray_search_radius_vector3d(self, branch, ray_o, ray_d, radius))
            return nearby_pts
        
        if not self._intersectAABB(self.root, ray_o, ray_d):
            return []
    
        return _ray_search_radius_vector3d(self, self.root, ray_o, ray_d, radius)
    
    def _intersectAABB(self, node, ray_o, ray_d):
        t_min = (node.lower - ray_o) / ray_d
        t_max = (node.upper - ray_o) / ray_d
        t_enter = np.minimum(t_min, t_max)
        t_exit = np.maximum(t_min, t_max)

        t_enter = np.maximum.reduce(t_enter)
        t_exit  = np.minimum.reduce(t_exit)
        return t_exit > t_enter and t_exit >= 0
    
    def _intersectAABB_withRadius(self, node, ray_o, ray_d, radius):
        t_min = (node.lower - radius - ray_o) / ray_d
        t_max = (node.upper + radius - ray_o) / ray_d
        t_enter = np.minimum(t_min, t_max)
        t_exit = np.maximum(t_min, t_max)

        t_enter = np.maximum.reduce(t_enter)
        t_exit  = np.minimum.reduce(t_exit)
        return t_exit > t_enter and t_exit >= 0

    def _intersect(self, node, position, radius):
        '''
            Compute the intersection between ball and cube
        '''
        d = 0.0
        for i in range(3):
            if position[i] < node.lower[i]:
                d += (position[i] - node.lower[i]) ** 2
            elif position[i] > node.upper[i]:
                d += (position[i] - node.upper[i]) ** 2
        return d <= radius ** 2

    def _findBranch(self, root, position):
        index = 0
        if position[0] >= root.position[0]:
            index |= 4
        if position[1] >= root.position[1]:
            index |= 2
        if position[2] >= root.position[2]:
            index |= 1
        return index
    
    def dist_point2ray(self, pt, ray_o, ray_d):
        ray_d = ray_d / np.linalg.norm(ray_d) # normalize the ray direction to unit vector
        sub_pt = pt - ray_o
        t = np.dot(sub_pt, ray_d)
        proj_pt = ray_o + t * ray_d
        return np.linalg.norm(proj_pt - pt)

def violent_enumeration(points, target, radius):
    nearby_pts = []
    for pt in points:
        if np.linalg.norm(np.array(pt.position) - np.array(target)) <= radius:
            nearby_pts.append(pt)
    return nearby_pts

def testPointSearching():
    # Number of objects we intend to add.
    NUM_TEST_OBJECTS = 2000

    # Number of lookups we're going to test
    NUM_LOOKUPS = 2000

    # Size that the octree covers
    WORLD_SIZE = 100.0

    #ORIGIN = (WORLD_SIZE, WORLD_SIZE, WORLD_SIZE)
    ORIGIN = (0, 0, 0)

    # The range from which to draw random values
    RAND_RANGE = (-WORLD_SIZE * 0.3, WORLD_SIZE * 0.3)

    # create random test objects
    testObjects = []
    for x in range(NUM_TEST_OBJECTS):
        the_name = "Node__" + str(x)
        the_pos = (
            ORIGIN[0] + random.randrange(*RAND_RANGE),
            ORIGIN[1] + random.randrange(*RAND_RANGE),
            ORIGIN[2] + random.randrange(*RAND_RANGE)
        )
        testObjects.append(Point(the_name, the_pos))

    # create some random positions to find as well
    findPositions = []
    for x in range(NUM_LOOKUPS):
        the_pos = (
            ORIGIN[0] + random.randrange(*RAND_RANGE),
            ORIGIN[1] + random.randrange(*RAND_RANGE),
            ORIGIN[2] + random.randrange(*RAND_RANGE)
        )
        findPositions.append(the_pos)
        
    
    octree = Octree(
        WORLD_SIZE,
        ORIGIN,
        10
    )
    t0 = time.time()
    for testObject in testObjects:
        octree.insertNode(testObject.position, testObject)
    t1 = time.time()
    print("Time to build octree: {} seconds".format(t1 - t0))
    
    for findPos in findPositions:
        t0 = time.time()
        nearby_pts = octree.pt_search_radius_vector3d(octree.root, findPos, 10)
        t1 = time.time()
        print("Time to find position: {} seconds".format(t1 - t0))
        print("Found {} nearby points".format(len(nearby_pts)))
        print(len(nearby_pts))
        print("-----")
        enumerated_nearby_pts = violent_enumeration(testObjects, findPos, 10)
        print(len(enumerated_nearby_pts))
        assert len(nearby_pts) == len(enumerated_nearby_pts)

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
    octree = Octree(leaf_data_size=10, points=pcd)
    t1 = time.time()
    print(f"build octree: {t1 - t0: .6f}s")
    '''
        Searching for points near a ray
    '''
    threshold = 0.016
    uv = [400, 400]
    ray_o = rays_o[uv[1], uv[0]].cpu().numpy()
    ray_d = rays_d[uv[1], uv[0]].cpu().numpy()
    t2 = time.time()
    nearby_poitns = octree.ray_search_radius_vector3d(octree.root, ray_o, ray_d, threshold)
    t3 = time.time()
    print(len(nearby_poitns))
    print(f"searching for points near a ray: {t3 - t2: .6f}s")
if __name__ == '__main__':
    
    '''
        testPoitSearching 
    '''
    # testPointSearching()

    '''
        testRaySearching
    '''
    testRaySearching()

    
