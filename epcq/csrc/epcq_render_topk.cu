#include <torch/extension.h>
#include <cub/cub.cuh>
#include "data_spec.hpp"
#include "data_spec_packed.cuh"
#include "cuda_util.cuh"
#include "render_util.cuh"
typedef cub::WarpReduce<float> WarpReduce;

namespace device{
    __global__ void quick_search_nearby_pc(PackedNeuralPointsSpec neural_points,
                                        PackedCameraSpec cam,
                                        RenderOptions options,
                                        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>sorted_point_idx_list,
                                        const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits>hash_table,
                                        torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> output_sp_xyz,
                                        torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits>mask_out,
                                        // Output
                                        torch::PackedTensorAccessor32<int, 4, torch::RestrictPtrTraits> output_pc_idx
                                        ){
    CUDA_GET_THREAD_ID(tid, cam.height * cam.width * options.num_sample_points_per_ray);
    const int ray_idx = tid / options.num_sample_points_per_ray;
    const int slot_idx = tid % options.num_sample_points_per_ray;
    const int ray_cnt_u = ray_idx % cam.width;
    const int ray_cnt_v = ray_idx / cam.width;
    if (!mask_out[ray_cnt_v][ray_cnt_u][slot_idx])
        return; // empty ray
    float sp_xyz[3] = {0.0f};
    #pragma unroll
    for (int i = 0; i < 3; i++)
        sp_xyz[i] = output_sp_xyz[ray_cnt_v][ray_cnt_u][slot_idx][i];

    int kernel_size2 = options.kernel_size * options.kernel_size;  
    UVSpec uv_list[POINT_BUFFER_SIZE];
    calculate_neighbor_uv_list(ray_cnt_v, ray_cnt_u, options.kernel_size, cam.height, cam.width, uv_list);
    PixSpec points_buffer[POINT_BUFFER_SIZE8];
    int points_buffer_size = 0;
    int max_dist_point_idx = 0;
    float max_dist = 1e3f;
    for (int i = 0; i < kernel_size2; i++){
        int v_valid = uv_list[i].v;
        int u_valid = uv_list[i].u;
        int buffer_size = hash_table[v_valid][u_valid][1];
        if (buffer_size < 0)
            continue;
        int start_idx = hash_table[v_valid][u_valid][0];
        int end_idx = start_idx + buffer_size - 1;
        for (int pc_idx = start_idx; pc_idx <= end_idx; pc_idx++){
            float pc_xyz[3] = {0.0f};
            neural_points.get_xyz_data(sorted_point_idx_list[pc_idx], pc_xyz);
            float dist = distance_between(sp_xyz, pc_xyz);
            if (dist < options.radius_threshold){
                if (points_buffer_size < POINT_BUFFER_SIZE8){
                    points_buffer[points_buffer_size] = PixSpec(dist, sorted_point_idx_list[pc_idx]);
                    if (max_dist > dist){
                        max_dist = dist;
                        max_dist_point_idx = points_buffer_size;
                    }
                    points_buffer_size++;
                }
                else{
                    // select sort
                    if (dist < max_dist){
                        points_buffer[max_dist_point_idx] = PixSpec(dist, sorted_point_idx_list[pc_idx]);
                        max_dist = dist;
                        for (int i = 0; i < POINT_BUFFER_SIZE8; i++){
                            if (points_buffer[i].dist > max_dist){
                                max_dist = points_buffer[i].dist;
                                max_dist_point_idx = i;
                            }
                        }
                    }
                }
            }
        }
        if (points_buffer_size > 0){
            for (int i = 0; i < points_buffer_size; i++){
                output_pc_idx[ray_cnt_v][ray_cnt_u][slot_idx][i] = points_buffer[i].idx;
            }
        }
    }
}
    __global__ void quick_sampling_kernel(PackedNeuralPointsSpec neural_points,
                                        PackedCameraSpec cam,
                                        RenderOptions options,
                                        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>sorted_point_idx_list,
                                        const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits>hash_table,
                                        // Output
                                        // [h, w, n, 3] sp_xyz
                                        torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> output_sp_xyz,
                                        // [h, w, n] mask_out
                                        torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits>mask_out,
                                        // [h, w, n] output_sp_t
                                        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>output_sp_t
                                        ){
    CUDA_GET_THREAD_ID(tid, cam.height * cam.width * WARP_SIZE);
    const int ray_idx = tid >> 5; // tid / WARP_SIZE
    const int ray_blk_idx = threadIdx.x >> 5;
    const int lane_idx = threadIdx.x & 0x1f;  
    int kernel_size2 = options.kernel_size * options.kernel_size;   

    // shared memory
    __shared__ SingleRaySpec warp_ray[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReduce::TempStorage comparer[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ float sample_point_buffer[TRACE_RAY_CUDA_RAYS_PER_BLOCK][MAX_SAMPLE_POINTS_NUM];
    __shared__ float sample_point_buffer_size[TRACE_RAY_CUDA_RAYS_PER_BLOCK][MAX_SAMPLE_POINTS_NUM];
   
    warp_ray[ray_blk_idx].tmin = 1e3f;
    warp_ray[ray_blk_idx].tmax = -1e3f;

    const int ray_cnt_u = ray_idx % cam.width;
    const int ray_cnt_v = ray_idx / cam.width;
    
    for (int i = 0; i < MAX_SAMPLE_POINTS_NUM / WARP_SIZE; i ++){
        sample_point_buffer[ray_blk_idx][i * WARP_SIZE + lane_idx] = 0.0f;
        sample_point_buffer_size[ray_blk_idx][i * WARP_SIZE + lane_idx] = 0.0f;
    }
    
    // Step1: find the valid uv buffer and calculate the tmin and tmax for each ray 
    if (lane_idx == 0){
        cam2world_ray(ray_cnt_u, ray_cnt_v, cam, warp_ray[ray_blk_idx].origin, warp_ray[ray_blk_idx].dir);
    }
    
    int v_valid = ray_cnt_v;
    int u_valid = ray_cnt_u;
    if (lane_idx < kernel_size2)
        calculate_neighbor_uv(ray_cnt_v, ray_cnt_u, options.kernel_size, cam.height, cam.width, lane_idx, v_valid, u_valid);

    int start_idx = hash_table[v_valid][u_valid][0];
    int buffer_size = hash_table[v_valid][u_valid][1];
    int end_idx = start_idx + buffer_size - 1;
    
    float tmax = -1e3f;
    float tmin = 1e3f;
    // store valid points in shared memory
    Projection proj_buffer[POINT_BUFFER_SIZE48];
    int proj_buffer_size = 0;
    
    if (lane_idx < kernel_size2 && buffer_size > 0){
        for (int pc_idx = start_idx; pc_idx <= end_idx && proj_buffer_size < POINT_BUFFER_SIZE48; pc_idx++){
            Projection proj; 
            float pc_xyz[3] = {0.0f};
            neural_points.get_xyz_data(sorted_point_idx_list[pc_idx], pc_xyz);
            warp_ray[ray_blk_idx].PointRayInterset(pc_xyz, proj);
            if (proj.dist_p2r < options.radius_threshold){
                if (tmin > proj.t){
                    tmin = proj.t;
                }
                if (tmax < proj.t){
                    tmax = proj.t;
                }
                proj_buffer[proj_buffer_size] = proj;
                proj_buffer_size++;
            }
        }
    }
    tmin = WarpReduce(comparer[ray_blk_idx]).Reduce(tmin, cub::Min());
    tmax = WarpReduce(comparer[ray_blk_idx]).Reduce(tmax, cub::Max());
    if (lane_idx == 0){
        warp_ray[ray_blk_idx].tmin = tmin;
        warp_ray[ray_blk_idx].tmax = tmax;
    }
    __syncwarp();
    
    if (warp_ray[ray_blk_idx].tmin > warp_ray[ray_blk_idx].tmax)
        return; // empty ray
    
    // coarse to fine sampling
    for(int i = 0; i < proj_buffer_size; i++){
        // calculate the slot index of projection point
        int slot_idx = (proj_buffer[i].t - warp_ray[ray_blk_idx].tmin) / options.t_intvl;
        if (slot_idx >= MAX_SAMPLE_POINTS_NUM)
            continue;
        atomicAdd(&sample_point_buffer[ray_blk_idx][slot_idx], proj_buffer[i].dist_p2r);
        atomicAdd(&sample_point_buffer_size[ray_blk_idx][slot_idx], 1.0f);
    }
    __syncwarp();
    for (int i = 0; i < MAX_SAMPLE_POINTS_NUM / WARP_SIZE; i ++){
        sample_point_buffer[ray_blk_idx][i * WARP_SIZE + lane_idx] /= (sample_point_buffer_size[ray_blk_idx][i * WARP_SIZE + lane_idx]+1e-6f);
    }

    if (lane_idx == 0){
        float transmit = 1.0f;
        int slot_idx = 0;
        for (int t_idx = 0; t_idx < MAX_SAMPLE_POINTS_NUM; t_idx++){
            // strategy1
            if (sample_point_buffer_size[ray_blk_idx][t_idx] < 2.0f)
                continue;
            if (transmit < 1e-3f) break;
            if (slot_idx >= output_sp_t.size(2)) break;
            float sdf = sample_point_buffer[ray_blk_idx][t_idx];
            float alpha = options.sdf2weight_gaussian_gamma * _EXP(- sdf / options.sdf2weight_gaussian_alpha);
            
            float weight = alpha * transmit;
            if (weight > 1e-3f){
                float t = warp_ray[ray_blk_idx].tmin + t_idx * options.t_intvl;
                float sp_xyz[3] = {0.0f};
                warp_ray[ray_blk_idx].ray_tracing(sp_xyz, t);
                for (int j = 0; j < 3; j++)
                    output_sp_xyz[ray_cnt_v][ray_cnt_u][slot_idx][j] = sp_xyz[j];
                mask_out[ray_cnt_v][ray_cnt_u][slot_idx] = true;
                output_sp_t[ray_cnt_v][ray_cnt_u][slot_idx] = t;
                slot_idx++;
            }
            transmit -= alpha;

            // strategy 2
            // if (sample_point_buffer_size[ray_blk_idx][t_idx] >= 2.0f){
            //     if (slot_idx >= output_sp_t.size(2)) break;
            //     float t = warp_ray[ray_blk_idx].tmin + t_idx * options.t_intvl;
            //     float sp_xyz[3] = {0.0f};
            //     warp_ray[ray_blk_idx].ray_tracing(sp_xyz, t);
            //     for (int j = 0; j < 3; j++)
            //         output_sp_xyz[ray_cnt_v][ray_cnt_u][slot_idx][j] = sp_xyz[j];
            //     mask_out[ray_cnt_v][ray_cnt_u][slot_idx] = true;
            //     output_sp_t[ray_cnt_v][ray_cnt_u][slot_idx] = t;
            //     slot_idx++;
            // }
        }
    }
}

__global__ void hashtable_query_for_nearby_point(PackedNeuralPointsSpec neural_points,
                                        PackedCameraSpec cam,
                                        RenderOptions options,
                                        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>sorted_point_idx_list,
                                        const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits>hash_table,
                                        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>indices_of_ray,
                                        // Output
                                        // [n, k]
                                        torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>output_pc_num,
                                        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> output_pc_idx
                                        ){
    const int kernel_size2 = options.kernel_size * options.kernel_size;
    const int necessary_thread_per_ray = ((kernel_size2 + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    const int num_ray = indices_of_ray.size(0);
    CUDA_GET_THREAD_ID(tid, necessary_thread_per_ray * num_ray);
    int blk_idx = tid / necessary_thread_per_ray;
    int lane_idx = tid % necessary_thread_per_ray;
    
    const int ray_idx = indices_of_ray[blk_idx];
    const int ray_cnt_u = ray_idx % cam.width;
    const int ray_cnt_v = ray_idx / cam.width;
    
    SingleRaySpec cur_ray;
    cam2world_ray(ray_cnt_u, ray_cnt_v, cam, cur_ray.origin, cur_ray.dir);
    int v_valid = ray_cnt_v;
    int u_valid = ray_cnt_u;
    
    calculate_neighbor_uv(ray_cnt_v, ray_cnt_u, options.kernel_size, cam.height, cam.width, lane_idx, v_valid, u_valid);
    int start_idx = hash_table[v_valid][u_valid][0];
    int buffer_size = hash_table[v_valid][u_valid][1];
    if (buffer_size < 0)
        return;
    int end_idx = start_idx + buffer_size - 1;

    int point_buffer[POINT_BUFFER_SIZE] = {-1};
    int point_idx = 0;
    if (lane_idx < kernel_size2){
        for (int pc_idx = start_idx; pc_idx <= end_idx && point_idx < POINT_BUFFER_SIZE; pc_idx++){
            Projection proj; 
            float pc_xyz[3] = {0.0f};
            neural_points.get_xyz_data(sorted_point_idx_list[pc_idx], pc_xyz);
            cur_ray.PointRayInterset(pc_xyz, proj);
            if (proj.dist_p2r < options.radius_threshold){
                point_buffer[point_idx] = sorted_point_idx_list[pc_idx];
                point_idx ++;   
            }
        }
    }
    __syncthreads();

    if (lane_idx < kernel_size2){
        // multi-thread write pc_idx into output_pc_idx using atomicAdd
        int cur_point_idx = atomicAdd(&output_pc_num[blk_idx], point_idx);
        if (cur_point_idx >= output_pc_idx.size(1)){
            return;
        }
        for (int i = cur_point_idx; i < cur_point_idx + point_idx; i++){
            if ( i >= output_pc_idx.size(1)) return ;
            output_pc_idx[blk_idx][i] = point_buffer[i - cur_point_idx];
        }
    }
}

}//namespace device

torch::Tensor quick_query_for_nearby_point(NeuralPointsSpec& neural_points, 
                                           CameraSpec& cam,
                                           RenderOptions& options,
                                           torch::Tensor& sorted_point_idx_list,
                                           torch::Tensor& hashtable,
                                           torch::Tensor& indices_of_ray
                                           ){
    neural_points.check();
    cam.check();
    CHECK_INPUT(sorted_point_idx_list);
    CHECK_INPUT(hashtable);     
    CHECK_INPUT(indices_of_ray);   
    if (hashtable.dim() == 2){
        hashtable = hashtable.view({cam.height, cam.width, 2});
    }
    if (hashtable.dim() != 3 || hashtable.size(0) != cam.height || hashtable.size(1) != cam.width || hashtable.size(2) != 2){
        printf("hashtable size: %d, %d, %d\n", static_cast<int>(hashtable.size(0)), static_cast<int>(hashtable.size(1)), static_cast<int>(hashtable.size(2)));
        throw std::runtime_error("hashtable size error");
    }        
    torch::TensorOptions intOptions = torch::TensorOptions()
                                               .dtype(torch::kInt32)
                                               .device(sorted_point_idx_list.device());
    
    int num_ray = indices_of_ray.size(0);
    torch::Tensor output_pc_idx = torch::full({num_ray, options.num_point_per_ray}, -1, intOptions);
    torch::Tensor output_pc_num = torch::zeros({num_ray}, intOptions);
    const int kernel_size2 = options.kernel_size * options.kernel_size;
    const int necessary_thread_per_ray = ((kernel_size2 + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    
    
    int num_threads = TRACE_RAY_CUDA_THREADS;
    if (num_threads < necessary_thread_per_ray){
        num_threads = (necessary_thread_per_ray + num_threads - 1) / num_threads * num_threads;
    }
    if (num_threads >= 1024){
        throw std::runtime_error("kernel size is too large, can not be handled by current CUDA kernel");
    }
    int num_ray_per_block = num_threads / necessary_thread_per_ray;
    int num_block = CUDA_N_BLOCKS_NEEDED(num_ray, num_ray_per_block);
    
    device::hashtable_query_for_nearby_point<<<num_block, num_threads>>>(neural_points,
                                                        cam,
                                                        options,
                                                        sorted_point_idx_list.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                        hashtable.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
                                                        indices_of_ray.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                        output_pc_num.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                        output_pc_idx.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
                                                        );
    CUDA_CHECK_ERRORS;                                                         
    return output_pc_idx;

}
// volume_render_epcq_image
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> quick_sampling(NeuralPointsSpec& neural_points, 
                                                        CameraSpec& cam,
                                                        RenderOptions& options,
                                                        torch::Tensor& sorted_point_idx_list,
                                                        torch::Tensor& hashtable 
                                                        ){
    neural_points.check();
    cam.check();
    CHECK_INPUT(sorted_point_idx_list);
    CHECK_INPUT(hashtable);
    if (hashtable.dim() == 2){
        hashtable = hashtable.view({cam.height, cam.width, 2});
    }
    if (hashtable.dim() != 3 || hashtable.size(0) != cam.height || hashtable.size(1) != cam.width || hashtable.size(2) != 2){
        printf("hashtable size: %d, %d, %d\n", static_cast<int>(hashtable.size(0)), static_cast<int>(hashtable.size(1)), static_cast<int>(hashtable.size(2)));
        throw std::runtime_error("hashtable size error");
    }
    
    torch::TensorOptions floatOptions = torch::TensorOptions()
                                               .dtype(torch::kFloat32)
                                               .device(sorted_point_idx_list.device());
    
    torch::TensorOptions intOptions = torch::TensorOptions()
                                               .dtype(torch::kInt32)
                                               .device(sorted_point_idx_list.device());
    
    torch::TensorOptions boolOptions = torch::TensorOptions()
                                               .dtype(torch::kBool)
                                               .device(sorted_point_idx_list.device());

    
    torch::Tensor output_sp_xyz = torch::zeros({cam.height, cam.width, options.num_sample_points_per_ray, 3}, floatOptions);
    torch::Tensor output_sp_t = torch::zeros({cam.height, cam.width, options.num_sample_points_per_ray}, floatOptions);
    torch::Tensor output_pc_idx = torch::full({cam.height, cam.width, options.num_sample_points_per_ray, options.num_point_cloud_per_sp}, -1, intOptions);
    torch::Tensor mask_out = torch::zeros({cam.height, cam.width, options.num_sample_points_per_ray}, boolOptions);
    
    {
        const int num_elem = cam.height * cam.width;
        const int num_threads = TRACE_RAY_CUDA_THREADS;
        const int num_blocks = CUDA_N_BLOCKS_NEEDED(num_elem * WARP_SIZE, num_threads);
        device::quick_sampling_kernel<<<num_blocks, num_threads>>>(neural_points,
                                                            cam,
                                                            options,
                                                            sorted_point_idx_list.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                            hashtable.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
                                                            //Output
                                                            output_sp_xyz.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                            mask_out.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
                                                            output_sp_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>()
                                                            );

        CUDA_CHECK_ERRORS;                                             
    }
    
    {
        const int num_elem = cam.height * cam.width * options.num_sample_points_per_ray; 
        
        int num_threads = TRACE_RAY_CUDA_THREADS;
        if (num_threads % options.num_sample_points_per_ray != 0){
            num_threads = CUDA_N_THREADS_NEEDED(TRACE_RAY_CUDA_THREADS, options.num_sample_points_per_ray) * options.num_sample_points_per_ray;
        }
        const int num_blocks = CUDA_N_BLOCKS_NEEDED(num_elem, num_threads);
        device::quick_search_nearby_pc<<<num_blocks, num_threads>>>(neural_points,
                                                            cam,
                                                            options,
                                                            sorted_point_idx_list.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                            hashtable.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
                                                            output_sp_xyz.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                                                            mask_out.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
                                                            //Output
                                                            output_pc_idx.packed_accessor32<int, 4, torch::RestrictPtrTraits>()
                                                            );

        CUDA_CHECK_ERRORS; 
    }
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>{output_sp_xyz, output_pc_idx, output_sp_t, mask_out};
}