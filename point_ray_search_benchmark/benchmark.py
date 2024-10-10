from plane import Plane, PlaneSearch, RenderOptions, PlaneSearchGPU, NeuralPoints
from kdtree import KDTreeRaySearch
from octree import Octree
from uniform_grid import UniformGrid
from tools import *
from pytorch_lightning.utilities import move_data_to_device
'''
Experiment 1:
    Fixed point cloud and change the number of ray
'''
def testMultiRaySearching(TEST_POINT_NUM=200000, TEST_RAY_NUM=np.arange(1000, 10000, 1000)):
    
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
    
    
    # pc_path = r'pointnerf_ckpt/nerf_synthesis/tiny_lego.ply'
    pc_path = r'pointnerf_ckpt/nerf_synthesis/lego.ply'
    
    pcd = o3d.io.read_point_cloud(pc_path)
    pcd = np.array(pcd.points)
    # shuffle pcd and select TEST_POINT_NUM points
    np.random.shuffle(pcd)
    pcd = pcd[:TEST_POINT_NUM]
    print("number of points: ", pcd.shape[0])
    
    radius = 0.016
    # single ray search
    # uv = [400, 400]
    # ray_o = rays_o[uv[1], uv[0]].cpu().numpy()
    # ray_d = rays_d[uv[1], uv[0]].cpu().numpy()
    
    # multiple ray search
    mask = data['valid_mask'].squeeze(-1).cpu().numpy()
    fg_uv_list = np.where(mask == 1)
    fg_uv_list = np.stack(fg_uv_list, axis=-1)
    for num_ray in TEST_RAY_NUM:
        np.random.shuffle(fg_uv_list)
        uv_list = fg_uv_list[:num_ray]
        
        result = {'num_ray':float(num_ray),
                'violent_enumeration':[],
                'uniform_grid':{'build':[], 'search':[]},
                'kdtree':{'build':[], 'search':[]},
                'octree':{'build':[], 'search':[]},
                'plane':{'build':[], 'search':[]}
                }
        '''
            Violent enumeration (for evaluation)
        '''
        # enu_t0 = time.time()
        # ray_o = rays_o[400, 400].cpu().numpy()
        # ray_d = rays_d[400, 400].cpu().numpy()
        # nearby_poitns, _ = violent_enumeration_ray(ray_o, ray_d, pcd, radius)
        # enu_t1 = time.time()
        # print(f'violent enumeration: {enu_t1 - enu_t0:.6f}s \t results: {len(nearby_poitns)} points')

        '''
            Uniform grid construction
        '''
        ug_t0 = time.time()
        voxel_size = 0.2
        uniform_grid = UniformGrid(pcd, voxel_size)
        ug_t1 = time.time()
        result['uniform_grid']['build'].append(ug_t1 - ug_t0)
        '''
            Uniform grid searching
        '''
        ug_search_time = 0
        progress_bar = tqdm(uv_list, unit="uv", desc='Uniform Grid')
        num_search = 1
        for uv in progress_bar:
            ray_o = rays_o[uv[1], uv[0]].cpu().numpy()
            ray_d = rays_d[uv[1], uv[0]].cpu().numpy()
            
            ug_t2 = time.time()
            nearby_poitns = uniform_grid.ray_search_radius_vector3d(ray_o, ray_d, radius)
            ug_t3 = time.time()
            ug_search_time += ug_t3 - ug_t2
            mean_ug_search_time = ug_search_time / num_search
            num_search += 1
            progress_bar.set_postfix({
                'build': f'{ug_t1 - ug_t0:.6f}s',
                'search': f'{mean_ug_search_time:.6f}s',
                # 'results': f'{len(nearby_poitns)} points'
            }, refresh=True)
            del nearby_poitns

        result['uniform_grid']['search'].append(ug_search_time)
        del uniform_grid

        '''
            Kdtree construction
        '''
        kdt_t0 = time.time()
        kdtree = KDTreeRaySearch(pcd, max_points_per_node=100)
        kdt_t1 = time.time()
        result['kdtree']['build'].append(kdt_t1 - kdt_t0)
        '''
            Kdtree searching
        '''
        kd_search_time = 0
        num_search = 1
        progress_bar = tqdm(uv_list, unit="uv", desc='Kdtree')
        for uv in progress_bar:
            ray_o = rays_o[uv[1], uv[0]].cpu().numpy()
            ray_d = rays_d[uv[1], uv[0]].cpu().numpy()

            kdt_t2 = time.time()
            nearby_poitns = kdtree.ray_search_radius_vector3d(kdtree.root, ray_o, ray_d, radius)
            kdt_t3 = time.time()
            kd_search_time += kdt_t3 - kdt_t2
            mean_kd_search_time = kd_search_time / num_search
            num_search += 1
            progress_bar.set_postfix({
                'build': f'{kdt_t1 - kdt_t0:.6f}s',
                'search': f'{mean_kd_search_time:.6f}s',
                # 'results': f'{len(nearby_poitns)} points'
            }, refresh=True)
            del nearby_poitns

        result['kdtree']['search'].append(kd_search_time)
        del kdtree
        
        '''
            Octree construction
        '''
        oct_t0 = time.time()
        octree = Octree(leaf_data_size=10, points=pcd)
        oct_t1 = time.time()
        result['octree']['build'].append(oct_t1 - oct_t0)
        '''
            Octree searching
        '''
        oct_search_time = 0
        num_search = 1
        progress_bar = tqdm(uv_list, unit="uv", desc='Octree')
        for uv in progress_bar:
            ray_o = rays_o[uv[1], uv[0]].cpu().numpy()
            ray_d = rays_d[uv[1], uv[0]].cpu().numpy()
            oct_t2 = time.time()
            nearby_poitns = octree.ray_search_radius_vector3d(ray_o, ray_d, radius)
            oct_t3 = time.time()
            oct_search_time += oct_t3 - oct_t2
            mean_oct_search_time = oct_search_time / num_search
            num_search += 1
            progress_bar.set_postfix({
                'build': f'{oct_t1 - oct_t0:.6f}s',
                'search': f'{mean_oct_search_time:.6f}s',
                # 'results': f'{len(nearby_poitns)} points'
            }, refresh=True)
            del nearby_poitns
        result['octree']['search'].append(oct_search_time)
        del octree

        '''
            Plane construction
        '''
        pln_t0 = time.time()
        plane = PlaneSearch(points=pcd, w2c=cam.w2c.cpu().numpy(), K=cam.K.cpu().numpy(), height=cam.height.cpu().numpy(), width=cam.width.cpu().numpy(), fx=cam.fx.cpu().numpy(), fy=cam.fy.cpu().numpy())
        pln_t1 = time.time()
        result['plane']['build'].append(pln_t1 - pln_t0)
        '''
            Plane searching
        '''
        pln_search_time = 0
        num_search = 1
        progress_bar = tqdm(uv_list, unit="uv", desc='Plane')
        for uv in progress_bar:
            ray_o = rays_o[uv[1], uv[0]].cpu().numpy()
            ray_d = rays_d[uv[1], uv[0]].cpu().numpy()
            pln_t2 = time.time()
            nearby_poitns = plane.ray_search_radius_vector3d(uv[0], uv[1], ray_o, ray_d, radius)
            pln_t3 = time.time()
            pln_search_time += pln_t3 - pln_t2
            mean_pln_search_time = pln_search_time / num_search
            num_search += 1
            progress_bar.set_postfix({
                'build': f'{pln_t1 - pln_t0:.6f}s',
                'search': f'{mean_pln_search_time:.6f}s',
                # 'results': f'{len(nearby_poitns)} points'
            }, refresh=True)
            del nearby_poitns
        result['plane']['search'].append(pln_search_time)
        del plane
        
        with open('point_ray_search_benchmark/benchmark.json', 'a') as f:
            json.dump(result, f)
            f.write('\n')

    
'''
Experiment 2:
    Fixed the number of ray and change the number of point cloud
'''
def testMultiPointSearching(TEST_POINT_NUM=np.arange(100000, 1000000, 100000), TEST_RAY_NUM=1000):
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
    
    
    # pc_path = r'pointnerf_ckpt/nerf_synthesis/tiny_lego.ply'
    pc_path = r'pointnerf_ckpt/nerf_synthesis/lego.ply'
    
    pcd = o3d.io.read_point_cloud(pc_path)
    pcd = np.array(pcd.points)
    # shuffle pcd and select TEST_POINT_NUM points
    np.random.shuffle(pcd)
    for num_point in TEST_POINT_NUM:
        if num_point < pcd.shape[0]:
            pcd = pcd[:num_point]
        else:
            # supplement points
            x_min = pcd[:, 0].min()
            x_max = pcd[:, 0].max()
            y_min = pcd[:, 1].min()
            y_max = pcd[:, 1].max()
            z_min = pcd[:, 2].min()
            z_max = pcd[:, 2].max()
            x = np.random.uniform(x_min, x_max, num_point - pcd.shape[0])
            y = np.random.uniform(y_min, y_max, num_point - pcd.shape[0])
            z = np.random.uniform(z_min, z_max, num_point - pcd.shape[0])
            pcd = np.concatenate([pcd, np.stack([x, y, z], axis=-1)], axis=0)

        print("number of points: ", pcd.shape[0], " number of rays: ", TEST_RAY_NUM)
        
        radius = 0.016
        # single ray search
        # uv = [400, 400]
        # ray_o = rays_o[uv[1], uv[0]].cpu().numpy()
        # ray_d = rays_d[uv[1], uv[0]].cpu().numpy()
        
        # multiple ray search
        mask = data['valid_mask'].squeeze(-1).cpu().numpy()
        fg_uv_list = np.where(mask == 1)
        fg_uv_list = np.stack(fg_uv_list, axis=-1)

        num_ray = TEST_RAY_NUM
        
        np.random.shuffle(fg_uv_list)
        uv_list = fg_uv_list[:num_ray]
        
        result = {'num_point':float(num_point),
                'num_ray':float(num_ray),
                'violent_enumeration':[],
                'uniform_grid':{'build':[], 'search':[]},
                'kdtree':{'build':[], 'search':[]},
                'octree':{'build':[], 'search':[]},
                'plane':{'build':[], 'search':[]}
                }
        '''
            Violent enumeration (for evaluation)
        '''
        # enu_t0 = time.time()
        # ray_o = rays_o[400, 400].cpu().numpy()
        # ray_d = rays_d[400, 400].cpu().numpy()
        # nearby_poitns, _ = violent_enumeration_ray(ray_o, ray_d, pcd, radius)
        # enu_t1 = time.time()
        # print(f'violent enumeration: {enu_t1 - enu_t0:.6f}s \t results: {len(nearby_poitns)} points')

        '''
            Uniform grid construction
        '''
        ug_t0 = time.time()
        voxel_size = 0.2
        uniform_grid = UniformGrid(pcd, voxel_size)
        ug_t1 = time.time()
        result['uniform_grid']['build'].append(ug_t1 - ug_t0)
        '''
            Uniform grid searching
        '''
        ug_search_time = 0
        progress_bar = tqdm(uv_list, unit="uv", desc='Uniform Grid')
        num_search = 1
        for uv in progress_bar:
            ray_o = rays_o[uv[1], uv[0]].cpu().numpy()
            ray_d = rays_d[uv[1], uv[0]].cpu().numpy()
            
            ug_t2 = time.time()
            nearby_poitns = uniform_grid.ray_search_radius_vector3d(ray_o, ray_d, radius)
            ug_t3 = time.time()
            ug_search_time += ug_t3 - ug_t2
            mean_ug_search_time = ug_search_time / num_search
            num_search += 1
            progress_bar.set_postfix({
                'build': f'{ug_t1 - ug_t0:.6f}s',
                'search': f'{mean_ug_search_time:.6f}s',
                # 'results': f'{len(nearby_poitns)} points'
            }, refresh=True)
            del nearby_poitns

        result['uniform_grid']['search'].append(ug_search_time)
        del uniform_grid

        '''
            Kdtree construction
        '''
        kdt_t0 = time.time()
        kdtree = KDTreeRaySearch(pcd, max_points_per_node=100)
        kdt_t1 = time.time()
        result['kdtree']['build'].append(kdt_t1 - kdt_t0)
        '''
            Kdtree searching
        '''
        kd_search_time = 0
        num_search = 1
        progress_bar = tqdm(uv_list, unit="uv", desc='Kdtree')
        for uv in progress_bar:
            ray_o = rays_o[uv[1], uv[0]].cpu().numpy()
            ray_d = rays_d[uv[1], uv[0]].cpu().numpy()

            kdt_t2 = time.time()
            nearby_poitns = kdtree.ray_search_radius_vector3d(kdtree.root, ray_o, ray_d, radius)
            kdt_t3 = time.time()
            kd_search_time += kdt_t3 - kdt_t2
            mean_kd_search_time = kd_search_time / num_search
            num_search += 1
            progress_bar.set_postfix({
                'build': f'{kdt_t1 - kdt_t0:.6f}s',
                'search': f'{mean_kd_search_time:.6f}s',
                # 'results': f'{len(nearby_poitns)} points'
            }, refresh=True)
            del nearby_poitns

        result['kdtree']['search'].append(kd_search_time)
        del kdtree
        
        '''
            Octree construction
        '''
        oct_t0 = time.time()
        octree = Octree(leaf_data_size=10, points=pcd)
        oct_t1 = time.time()
        result['octree']['build'].append(oct_t1 - oct_t0)
        '''
            Octree searching
        '''
        oct_search_time = 0
        num_search = 1
        progress_bar = tqdm(uv_list, unit="uv", desc='Octree')
        for uv in progress_bar:
            ray_o = rays_o[uv[1], uv[0]].cpu().numpy()
            ray_d = rays_d[uv[1], uv[0]].cpu().numpy()
            oct_t2 = time.time()
            nearby_poitns = octree.ray_search_radius_vector3d(ray_o, ray_d, radius)
            oct_t3 = time.time()
            oct_search_time += oct_t3 - oct_t2
            mean_oct_search_time = oct_search_time / num_search
            num_search += 1
            progress_bar.set_postfix({
                'build': f'{oct_t1 - oct_t0:.6f}s',
                'search': f'{mean_oct_search_time:.6f}s',
                # 'results': f'{len(nearby_poitns)} points'
            }, refresh=True)
            del nearby_poitns
        result['octree']['search'].append(oct_search_time)
        del octree

        '''
            Plane construction
        '''
        pln_t0 = time.time()
        plane = PlaneSearch(points=pcd, w2c=cam.w2c.cpu().numpy(), K=cam.K.cpu().numpy(), height=cam.height.cpu().numpy(), width=cam.width.cpu().numpy(), fx=cam.fx.cpu().numpy(), fy=cam.fy.cpu().numpy())
        pln_t1 = time.time()
        result['plane']['build'].append(pln_t1 - pln_t0)
        '''
            Plane searching
        '''
        pln_search_time = 0
        num_search = 1
        progress_bar = tqdm(uv_list, unit="uv", desc='Plane')
        for uv in progress_bar:
            ray_o = rays_o[uv[1], uv[0]].cpu().numpy()
            ray_d = rays_d[uv[1], uv[0]].cpu().numpy()
            pln_t2 = time.time()
            nearby_poitns = plane.ray_search_radius_vector3d(uv[0], uv[1], ray_o, ray_d, radius)
            pln_t3 = time.time()
            pln_search_time += pln_t3 - pln_t2
            mean_pln_search_time = pln_search_time / num_search
            num_search += 1
            progress_bar.set_postfix({
                'build': f'{pln_t1 - pln_t0:.6f}s',
                'search': f'{mean_pln_search_time:.6f}s',
                # 'results': f'{len(nearby_poitns)} points'
            }, refresh=True)
            del nearby_poitns
        result['plane']['search'].append(pln_search_time)
        del plane
        
        with open('point_ray_search_benchmark/benchmark1.json', 'a') as f:
            json.dump(result, f)
            f.write('\n')

def testMultiRaySearchingPlane_cpu_gpu(TEST_POINT_NUM=200000, TEST_RAY_NUM=np.arange(1000, 10000, 1000)):
    args = parser.parse_args()
    status = 'inference'#'train_on_pc'
    nerfsynthesis = NeRFSynthesis(args, 'lego', 
                                  ref_downsample=1,
                                  src_downsample=1,
                                  split='train',
                                  status=status)
    data_cpu = nerfsynthesis[0]
    data_gpu = move_data_to_device(nerfsynthesis[0], torch.device(args.device))
    fx, fy, cx, cy = CameraSpec.decomposeK(data_cpu['ref_K'])
    cam_cpu = CameraSpec(c2w=data_cpu['ref_c2w'],
                        w2c=data_cpu['ref_w2c'],
                        K = data_cpu['ref_K'],
                        fx=fx, fy=fy,
                        cx=cx, cy=cy,
                        height=data_cpu['ref_hw'][0],
                        width=data_cpu['ref_hw'][1])
    rays_o, rays_d = generate_rays(cam_cpu)
    cam_gpu = CameraSpec(c2w=data_gpu['ref_c2w'],
                        w2c=data_gpu['ref_w2c'],
                        K = data_gpu['ref_K'],
                        fx=fx, fy=fy,
                        cx=cx, cy=cy,
                        height=data_gpu['ref_hw'][0],
                        width=data_gpu['ref_hw'][1])
    
    pc_path = r'pointnerf_ckpt/nerf_synthesis/lego.ply'
    
    options = RenderOptions()
    
    pcd = o3d.io.read_point_cloud(pc_path)
    pcd = np.array(pcd.points).astype(np.float32)
    # shuffle pcd and select TEST_POINT_NUM points
    np.random.shuffle(pcd)
    pcd = pcd[:TEST_POINT_NUM]
    print("number of points: ", pcd.shape[0])
    pcd_gpu = torch.from_numpy(pcd).cuda()
    
    radius = 0.016
    mask = data_cpu['valid_mask'].squeeze(-1).cpu().numpy()
    fg_uv_list = np.where(mask == 1)
    fg_uv_list = np.stack(fg_uv_list, axis=-1)
    
    for num_ray in TEST_RAY_NUM:
        np.random.shuffle(fg_uv_list)
        uv_list = fg_uv_list[:num_ray]
        result = {'num_ray':float(num_ray),
                 'plane_cpu':{'build':[], 'search':[]},
                 'plane_gpu':{'build':[], 'search':[]}
                }
        
        '''
            Plane construction cpu
        '''
        pln_t0 = time.time()
        plane = PlaneSearch(points=pcd, w2c=cam_cpu.w2c.cpu().numpy(), K=cam_cpu.K.cpu().numpy(), height=cam_cpu.height.cpu().numpy(), width=cam_cpu.width.cpu().numpy(), fx=cam_cpu.fx.cpu().numpy(), fy=cam_cpu.fy.cpu().numpy())
        pln_t1 = time.time()
        result['plane_cpu']['build'].append(pln_t1 - pln_t0)
        '''
            Plane searching cpu
        '''
        pln_search_time = 0
        num_search = 1
        progress_bar = tqdm(uv_list, unit="uv", desc='Plane')
        for uv in progress_bar:
            ray_o = rays_o[uv[1], uv[0]].cpu().numpy()
            ray_d = rays_d[uv[1], uv[0]].cpu().numpy()
            pln_t2 = time.time()
            nearby_poitns = plane.ray_search_radius_vector3d(uv[0], uv[1], ray_o, ray_d, radius)
            pln_t3 = time.time()
            pln_search_time += pln_t3 - pln_t2
            mean_pln_search_time = pln_search_time / num_search
            num_search += 1
            progress_bar.set_postfix({
                'build': f'{pln_t1 - pln_t0:.6f}s',
                'search': f'{mean_pln_search_time:.6f}s',
                # 'results': f'{len(nearby_poitns)} points'
            }, refresh=True)
            del nearby_poitns
        result['plane_cpu']['search'].append(pln_search_time)
        del plane
        
        '''
            Plane construction gpu
        '''
        pln_gpu_t0 = time.time()
        plane = PlaneSearchGPU()
        sorted_point_idx_list, sorted_depth_list, hash_table = plane.build_data_structure(pcd_gpu, cam_gpu, sort=False)
        torch.cuda.synchronize()
        neural_points = NeuralPoints(pc_type='pc', points = pcd_gpu)
        pln_gpu_t1 = time.time()
        result['plane_gpu']['build'].append(pln_gpu_t1 - pln_gpu_t0)
        '''
            Plane searching gpu
        '''
        pln_gpu_t2 = time.time()
        output_pc_idx = plane.query(neural_points, cam_gpu, options, sorted_point_idx_list, hash_table, torch.tensor(uv_list).cuda())
        torch.cuda.synchronize()
        pln_gpu_t3 = time.time()
        result['plane_gpu']['search'].append(pln_gpu_t3 - pln_gpu_t2)
        print("plane gpu build time: ", pln_gpu_t1 - pln_gpu_t0, " plane gpu search time: ", pln_gpu_t3 - pln_gpu_t2)
        with open('point_ray_search_benchmark/benchmark_plane_cpu_gpu.json', 'a') as f:
            json.dump(result, f)
            f.write('\n')
        
def testMultiPointSearchingPlane_cpu_gpu(TEST_POINT_NUM=np.arange(100000, 1000000, 100000), TEST_RAY_NUM=1000):
    args = parser.parse_args()
    status = 'inference'#'train_on_pc'
    nerfsynthesis = NeRFSynthesis(args, 'lego', 
                                ref_downsample=1,
                                src_downsample=1,
                                split='train',
                                status=status)
    data_cpu = nerfsynthesis[0]
    data_gpu = move_data_to_device(nerfsynthesis[0], torch.device(args.device))
    fx, fy, cx, cy = CameraSpec.decomposeK(data_cpu['ref_K'])
    cam_cpu = CameraSpec(c2w=data_cpu['ref_c2w'],
                        w2c=data_cpu['ref_w2c'],
                        K = data_cpu['ref_K'],
                        fx=fx, fy=fy,
                        cx=cx, cy=cy,
                        height=data_cpu['ref_hw'][0],
                        width=data_cpu['ref_hw'][1])
    rays_o, rays_d = generate_rays(cam_cpu)
    cam_gpu = CameraSpec(c2w=data_gpu['ref_c2w'],
                        w2c=data_gpu['ref_w2c'],
                        K = data_gpu['ref_K'],
                        fx=fx, fy=fy,
                        cx=cx, cy=cy,
                        height=data_gpu['ref_hw'][0],
                        width=data_gpu['ref_hw'][1])
    
    pc_path = r'pointnerf_ckpt/nerf_synthesis/lego.ply'
    
    options = RenderOptions()
    
    pcd = o3d.io.read_point_cloud(pc_path)
    pcd = np.array(pcd.points).astype(np.float32)
    # shuffle pcd and select TEST_POINT_NUM points
    radius = 0.016
    mask = data_cpu['valid_mask'].squeeze(-1).cpu().numpy()
    fg_uv_list = np.where(mask == 1)
    fg_uv_list = np.stack(fg_uv_list, axis=-1)
    
    for num_point in TEST_POINT_NUM:
        np.random.shuffle(pcd)
        if num_point < pcd.shape[0]:
            pcd_ = pcd[:num_point]
        else:
            # supplement points
            x_min = pcd[:, 0].min()
            x_max = pcd[:, 0].max()
            y_min = pcd[:, 1].min()
            y_max = pcd[:, 1].max()
            z_min = pcd[:, 2].min()
            z_max = pcd[:, 2].max()
            x = np.random.uniform(x_min, x_max, num_point - pcd.shape[0])
            y = np.random.uniform(y_min, y_max, num_point - pcd.shape[0])
            z = np.random.uniform(z_min, z_max, num_point - pcd.shape[0])
            pcd_ = np.concatenate([pcd, np.stack([x, y, z], axis=-1)], axis=0)
        print("number of points: ", pcd_.shape[0])
        pcd_gpu = torch.from_numpy(pcd_).cuda()
        
        num_ray = TEST_RAY_NUM
        np.random.shuffle(fg_uv_list)
        uv_list = fg_uv_list[:num_ray]
        result = {
                'num_points':pcd_.shape[0],
                'num_ray':float(num_ray),
                'plane_cpu':{'build':[], 'search':[]},
                'plane_gpu':{'build':[], 'search':[]}
                }
        
        '''
            Plane construction cpu
        '''
        pln_t0 = time.time()
        plane = PlaneSearch(points=pcd_, w2c=cam_cpu.w2c.cpu().numpy(), K=cam_cpu.K.cpu().numpy(), height=cam_cpu.height.cpu().numpy(), width=cam_cpu.width.cpu().numpy(), fx=cam_cpu.fx.cpu().numpy(), fy=cam_cpu.fy.cpu().numpy())
        pln_t1 = time.time()
        result['plane_cpu']['build'].append(pln_t1 - pln_t0)
        '''
            Plane searching cpu
        '''
        pln_search_time = 0
        num_search = 1
        progress_bar = tqdm(uv_list, unit="uv", desc='Plane')
        for uv in progress_bar:
            ray_o = rays_o[uv[1], uv[0]].cpu().numpy()
            ray_d = rays_d[uv[1], uv[0]].cpu().numpy()
            pln_t2 = time.time()
            nearby_poitns = plane.ray_search_radius_vector3d(uv[0], uv[1], ray_o, ray_d, radius)
            pln_t3 = time.time()
            pln_search_time += pln_t3 - pln_t2
            mean_pln_search_time = pln_search_time / num_search
            num_search += 1
            progress_bar.set_postfix({
                'build': f'{pln_t1 - pln_t0:.6f}s',
                'search': f'{mean_pln_search_time:.6f}s',
                # 'results': f'{len(nearby_poitns)} points'
            }, refresh=True)
            del nearby_poitns
        result['plane_cpu']['search'].append(pln_search_time)
        del plane
        
        '''
            Plane construction gpu
        '''
        pln_gpu_t0 = time.time()
        plane = PlaneSearchGPU()
        sorted_point_idx_list, sorted_depth_list, hash_table = plane.build_data_structure(pcd_gpu, cam_gpu, sort=False)
        torch.cuda.synchronize()
        neural_points = NeuralPoints(pc_type='pc', points = pcd_gpu)
        pln_gpu_t1 = time.time()
        result['plane_gpu']['build'].append(pln_gpu_t1 - pln_gpu_t0)
        '''
            Plane searching gpu
        '''
        pln_gpu_t2 = time.time()
        output_pc_idx = plane.query(neural_points, cam_gpu, options, sorted_point_idx_list, hash_table, torch.tensor(uv_list).cuda())
        torch.cuda.synchronize()
        pln_gpu_t3 = time.time()
        result['plane_gpu']['search'].append(pln_gpu_t3 - pln_gpu_t2)
        print("plane gpu build time: ", pln_gpu_t1 - pln_gpu_t0, " plane gpu search time: ", pln_gpu_t3 - pln_gpu_t2)
        with open('point_ray_search_benchmark/benchmark_point_plane_cpu_gpu.json', 'a') as f:
            json.dump(result, f)
            f.write('\n')
            
if __name__ == '__main__':
    # testMultiRaySearching()
    # testMultiPointSearching()
    # for _ in range(5):
    #     testMultiRaySearchingPlane_cpu_gpu()
    
    for _ in range(5):
        testMultiPointSearchingPlane_cpu_gpu()