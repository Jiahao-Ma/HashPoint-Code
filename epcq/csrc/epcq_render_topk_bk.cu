// #include <torch/extension.h>
// #include <cub/cub.cuh>
// #include "render_util.cuh"
// #include "data_spec.hpp"
// #include "data_spec_packed.cuh"
// #include "cuda_util.cuh"
// typedef cub::WarpReduce<float> WarpReduce;
// namespace device{

// __device__ void interpolate_topk_backward_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_data_sh,
//                                                  torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_data_sigma,
//                                                  torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> mask_out,
//                                                  const int grad_pc_index,
//                                                  const float grad_weight,
//                                                  const float& weight_sum,
//                                                  const int sh_size,
//                                                  float* cur_grad_sh,
//                                                  float cur_grad_sigma){
//     if (grad_pc_index < 0) return;                                                  
//     for (int sh_idx = 0; sh_idx < sh_size; sh_idx++){
//         atomicAdd(&grad_data_sh[grad_pc_index][sh_idx], (grad_weight / weight_sum) * cur_grad_sh[sh_idx]);
//     }
//     // calculate the gradient for sigma
//     atomicAdd(&grad_data_sigma[grad_pc_index][0], (grad_weight / weight_sum) * cur_grad_sigma);
//     mask_out[grad_pc_index] = true;
// }
// __global__ void render_image_backward_kernel_with_early_stop_topk(PackedNeuralPointsSpec neural_points,
//                                     PackedCameraSpec cam,
//                                     RenderOptions options,
//                                     const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>sorted_point_idx_list,
//                                     const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>sorted_depth_list,
//                                     const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits>hash_table,
//                                     torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> ray_idx_list,
//                                     torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rgb_out,
//                                     torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rgb_gt,
//                                     //Output
//                                     PackedNeuralPointsGrads grads
//                                     ){
//     const int ray_blk_idx = threadIdx.x >> 5;
//     const int lane_idx = threadIdx.x & 0x1f;
//     const int num_ray_per_block = blockDim.x >> 5;
//     const int bin_idx = blockIdx.x * num_ray_per_block + ray_blk_idx;
//     const int ray_idx = ray_idx_list[bin_idx];
//     int kernel_size2 = options.kernel_size * options.kernel_size;

//     __shared__ typename WarpReduce::TempStorage sigma_buffer[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
//     __shared__ typename WarpReduce::TempStorage sh_buffer[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
//     __shared__ typename WarpReduce::TempStorage weight_buffer[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
//     __shared__ SingleRaySpec warp_ray[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
//     __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
//     __shared__ float tmin_buffer[TRACE_RAY_CUDA_RAYS_PER_BLOCK][WARP_SIZE];
//     __shared__ float tmax_buffer[TRACE_RAY_CUDA_RAYS_PER_BLOCK][WARP_SIZE];
//     __shared__ PixSpec points_buffer[TRACE_RAY_CUDA_RAYS_PER_BLOCK][POINT_BUFFER_SIZE48];
//     __shared__ int points_buffer_size[TRACE_RAY_CUDA_RAYS_PER_BLOCK];

//     points_buffer_size[ray_blk_idx] = 0;
//     tmin_buffer[ray_blk_idx][lane_idx] = 1e3f;
//     tmax_buffer[ray_blk_idx][lane_idx] = -1e3f;
//     warp_ray[ray_blk_idx].tmin = 1e3f;
//     warp_ray[ray_blk_idx].tmax = -1e3f;

//     const int ray_cnt_u = ray_idx % cam.width;
//     const int ray_cnt_v = ray_idx / cam.width;
//     // --- START of forward pass --- //
//     // Step1: find the valid uv buffer and calculate the tmin and tmax for each ray 
//     if (lane_idx == 0){
//         cam2world_ray(ray_cnt_u, ray_cnt_v, cam, warp_ray[ray_blk_idx].origin, warp_ray[ray_blk_idx].dir);
//     }
//     int v_valid = ray_cnt_v;
//     int u_valid = ray_cnt_u;
//     calculate_neighbor_uv(ray_cnt_v, ray_cnt_u, options.kernel_size, cam.height, cam.width, lane_idx, v_valid, u_valid);
//     // if start_idx and buffer_size is -1, the ray is empty
//     int start_idx = hash_table[v_valid][u_valid][0];
//     int buffer_size = hash_table[v_valid][u_valid][1];
//     int end_idx = start_idx + buffer_size - 1;
//     if (lane_idx < kernel_size2 && buffer_size > 0)
//         preprocess_ray_kernel(neural_points, 
//                                 sorted_point_idx_list[start_idx], 
//                                 sorted_point_idx_list[end_idx],
//                                 warp_ray[ray_blk_idx],
//                                 tmin_buffer[ray_blk_idx][lane_idx],
//                                 tmax_buffer[ray_blk_idx][lane_idx]);
//     __syncthreads();
//     if (lane_idx == 0){
//         dump_min<float>(tmin_buffer[ray_blk_idx], warp_ray[ray_blk_idx].tmin, WARP_SIZE);
//         dump_max<float>(tmax_buffer[ray_blk_idx], warp_ray[ray_blk_idx].tmax, WARP_SIZE);
//     }
//     __syncthreads();
//     if (warp_ray[ray_blk_idx].tmin > warp_ray[ray_blk_idx].tmax){
//         #pragma unroll
//         for (int i = 0; i < 3; i++)
//             rgb_out[bin_idx][i] = options.background_brightness;
//         return; 
//     } // empty ray
//     if (lane_idx >= kernel_size2) return;
// // Step2: calculate sh 
//     if (lane_idx == 0)
//         calc_sh(neural_points.basis_dim, warp_ray[ray_blk_idx].dir, sphfunc_val[ray_blk_idx]);

//     // Step3: find the nearest K points for each sample point and aggregate feature 
//     float t = warp_ray[ray_blk_idx].tmin;   
//     float sp_xyz[3] = {0.0f, 0.0f, 0.0f};
//     float pc_xyz[3] = {0.0f, 0.0f, 0.0f};
//     float pred_color[3] = {0.0f, 0.0f, 0.0f};
    
//     float log_transmit = 0.0f;
//     while(t < warp_ray[ray_blk_idx].tmax){
//     // for (int sp_idx = 0; sp_idx < options.num_sample_points; sp_idx++){ // for debug
//         // Step3.1: find the neaby points for each ray
//         warp_ray[ray_blk_idx].ray_tracing(sp_xyz, t);
//         PixSpec tmp_point_buffer[POINT_BUFFER_SIZE4];
//         int tmp_point_buffer_size = 0;

//         if (buffer_size > 0){
//             // depth-based binary search 
//             int mid_index = binary_search_index(neural_points, sorted_point_idx_list, warp_ray[ray_blk_idx], start_idx, buffer_size, t);
            
//             int point_idx = sorted_point_idx_list[mid_index];
//             neural_points.get_xyz_data(point_idx, pc_xyz);
//             float dist = distance_between(sp_xyz, pc_xyz);
//             if (dist < options.radius_threshold){
//                 tmp_point_buffer[tmp_point_buffer_size] = PixSpec(dist, point_idx);
//                 tmp_point_buffer_size++;
//             }

//             int left_index = mid_index - 1;
//             while (left_index >= start_idx){
//                 point_idx = sorted_point_idx_list[left_index];
//                 neural_points.get_xyz_data(point_idx, pc_xyz);
//                 dist = distance_between(sp_xyz, pc_xyz);
//                 if (dist < options.radius_threshold){
//                     tmp_point_buffer[tmp_point_buffer_size] = PixSpec(dist, point_idx);
//                     tmp_point_buffer_size++;
//                     if (tmp_point_buffer_size >= POINT_BUFFER_SIZE4) break;
//                 }
//                 else{
//                     break;
//                 }
//                 left_index--;
//             }

//             int right_index = mid_index + 1;
//             while (right_index <= end_idx && tmp_point_buffer_size < POINT_BUFFER_SIZE4){
//                 point_idx = sorted_point_idx_list[right_index];
//                 neural_points.get_xyz_data(point_idx, pc_xyz);
//                 dist = distance_between(sp_xyz, pc_xyz);
//                 if (dist < options.radius_threshold){
//                     tmp_point_buffer[tmp_point_buffer_size] = PixSpec(dist, point_idx);
//                     tmp_point_buffer_size++;
//                     if (tmp_point_buffer_size >= POINT_BUFFER_SIZE4) break;
//                 }
//                 else{
//                     break;
//                 }
//                 right_index++;
//             }
//         }
//         __syncwarp((1U << kernel_size2) - 1);
//         int start_index = atomicAdd(&points_buffer_size[ray_blk_idx], tmp_point_buffer_size);
//         for (int i = 0; i < tmp_point_buffer_size; i++){
//             if (start_index + i >= POINT_BUFFER_SIZE48){
//                 points_buffer_size[ray_blk_idx] = POINT_BUFFER_SIZE48;
//                 break;
//             }
//             points_buffer[ray_blk_idx][start_index + i] = tmp_point_buffer[i];
//         }
//         // Step3.2: find the topK points for each ray
//         __syncwarp((1U << kernel_size2) - 1);
//         // if no points in the buffer, skip the following steps
//         float weight = 0.0f;
//         float sigma = 0.0f;
//         float sp_sh[SH_SIZE] = {0.0f};
//         float pcnt = 0.0f;
//         float color_weight = 0.0f;
//         if (points_buffer_size[ray_blk_idx] > 0){
//             if (lane_idx == 0){
//                 bubble_sort<PixSpec>(points_buffer[ray_blk_idx], points_buffer_size[ray_blk_idx], false);
//             }
//             __syncwarp((1U << kernel_size2) - 1);
//             // Step3.3: aggregate feature
//             // select topK points from sorted points_buffer
//             int valid_pc_num = min(options.topK, points_buffer_size[ray_blk_idx]);
//             if (lane_idx < valid_pc_num){
//                 float sdf = points_buffer[ray_blk_idx][lane_idx].dist;
//                 int point_idx = points_buffer[ray_blk_idx][lane_idx].idx;
//                 weight = sdf2weight(sdf, options);
//                 sigma = neural_points.get_density_data(point_idx);
//                 sigma = sigma * weight;
//                 neural_points.get_sh_data(point_idx, sp_sh);
//                 cuda_util::_fmulf_vector(sp_sh, weight, SH_SIZE);
//                 // printf("[%d %d] point_idx: %d  sdf: %d  sigma: %f\n", ray_cnt_v, ray_cnt_u, lane_idx, point_idx, sdf, sigma);
//             }

//             sigma = cub::WarpReduce<float>(sigma_buffer[ray_blk_idx]).Sum(sigma, kernel_size2);
//             weight = cub::WarpReduce<float>(weight_buffer[ray_blk_idx]).Sum(weight, kernel_size2);
//             weight = weight + 1e-8f;
//             sigma = fdividef(sigma, weight);
//             sigma = __shfl_sync((1U << kernel_size2) - 1, sigma, 0); // broadcast sigma to all threads (warp) for calculating the same log_transmit
//             // Step4: volume rendering 
//             pcnt = options.t_intvl * fmaxf(sigma, 0.0f); 
//             color_weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
//             log_transmit -= pcnt;
          
//             #pragma unroll 
//             for (int i = 0; i < SH_SIZE; i++){
//                 sp_sh[i] = WarpReduce(sh_buffer[ray_blk_idx]).Sum(sp_sh[i], kernel_size2);
//                 sp_sh[i] = fdividef(sp_sh[i], weight);
//             }
//             float color[3] = {0.0f, 0.0f, 0.0f};
//             #pragma unroll
//             for (int i = 0; i < neural_points.basis_dim; i++){
//                 color[0] += sp_sh[i + neural_points.basis_dim * 0] * sphfunc_val[ray_blk_idx][i];
//                 color[1] += sp_sh[i + neural_points.basis_dim * 1] * sphfunc_val[ray_blk_idx][i];
//                 color[2] += sp_sh[i + neural_points.basis_dim * 2] * sphfunc_val[ray_blk_idx][i];
//             }
//             pred_color[0] += color_weight * fmaxf(color[0] + 0.5f, 0.0f);
//             pred_color[1] += color_weight * fmaxf(color[1] + 0.5f, 0.0f);
//             pred_color[2] += color_weight * fmaxf(color[2] + 0.5f, 0.0f);
            
//             if (_EXP(log_transmit) < options.stop_threshold) break;
//         }
//         // if (lane_idx == 0){
//         //     if (points_buffer_size[ray_blk_idx] > 0){
//         //         int valid_pc_num = min(options.topK, points_buffer_size[ray_blk_idx]);
//         //         for (int i = 0; i < valid_pc_num; i++){
//         //             float sdf = points_buffer[ray_blk_idx][i].dist;
//         //             int point_idx = points_buffer[ray_blk_idx][i].idx;
//         //             printf("[%d][%d] point_idx: %d  sdf: %f\n", sp_idx, i, point_idx, sdf);
//         //         }
//         //     }
//         //     printf("[%d %d][%d] t: %f sigma: %f color_weight: %f  color: %f %f %f\n", ray_cnt_v, ray_cnt_u, sp_idx, t, sigma, color_weight, pred_color[0], pred_color[1], pred_color[2]);
//         // }
//         t += options.t_intvl;
//         points_buffer_size[ray_blk_idx] = 0;
//     }
//     if (lane_idx == 0){
//         #pragma unroll 3
//         for (int i = 0; i < 3; i++){
//             pred_color[i] += _EXP(log_transmit) * options.background_brightness;
//             rgb_out[bin_idx][i] = pred_color[i];
//         }
//     }
//     points_buffer_size[ray_blk_idx] = 0;
//     __syncwarp((1U << kernel_size2) - 1);
//     // --- END of forward pass --- //
//     // --- START of backward pass --- //
//     float grad_out[3] = {0.0f};
//     pred_color[0] = __shfl_sync((1U << kernel_size2) - 1, pred_color[0], 0);
//     pred_color[1] = __shfl_sync((1U << kernel_size2) - 1, pred_color[1], 0);
//     pred_color[2] = __shfl_sync((1U << kernel_size2) - 1, pred_color[2], 0);
//     const float norm_factor = 2.f / (3 * rgb_gt.size(0));
//     #pragma unroll 3
//     for (int i = 0; i < 3; i++){
//         float resid = rgb_out[bin_idx][i] - rgb_gt[bin_idx][i]; 
//         grad_out[i] = norm_factor * resid;
//     }
//     float accum = fmaf(pred_color[0], grad_out[0],
//                         fmaf(pred_color[1], grad_out[1],
//                              pred_color[2] * grad_out[2]));
//     t = warp_ray[ray_blk_idx].tmin;
//     log_transmit = 0.0f;

//     while(t < warp_ray[ray_blk_idx].tmax){
//     // for (int sp_idx = 0; sp_idx < options.num_sample_points; sp_idx++){ // for debug
//         // Step3.1: find the neaby points for each ray
//         warp_ray[ray_blk_idx].ray_tracing(sp_xyz, t);
//         PixSpec tmp_point_buffer[POINT_BUFFER_SIZE4];
//         int tmp_point_buffer_size = 0;

//         if (buffer_size > 0){
//             // depth-based binary search 
//             int mid_index = binary_search_index(neural_points, sorted_point_idx_list, warp_ray[ray_blk_idx], start_idx, buffer_size, t);
            
//             int point_idx = sorted_point_idx_list[mid_index];
//             neural_points.get_xyz_data(point_idx, pc_xyz);
//             float dist = distance_between(sp_xyz, pc_xyz);
//             if (dist < options.radius_threshold){
//                 tmp_point_buffer[tmp_point_buffer_size] = PixSpec(dist, point_idx);
//                 tmp_point_buffer_size++;
//             }

//             int left_index = mid_index - 1;
//             while (left_index >= start_idx){
//                 point_idx = sorted_point_idx_list[left_index];
//                 neural_points.get_xyz_data(point_idx, pc_xyz);
//                 dist = distance_between(sp_xyz, pc_xyz);
//                 if (dist < options.radius_threshold){
//                     tmp_point_buffer[tmp_point_buffer_size] = PixSpec(dist, point_idx);
//                     tmp_point_buffer_size++;
//                     if (tmp_point_buffer_size >= POINT_BUFFER_SIZE4) break;
//                 }
//                 else{
//                     break;
//                 }
//                 left_index--;
//             }

//             int right_index = mid_index + 1;
//             while (right_index <= end_idx && tmp_point_buffer_size < POINT_BUFFER_SIZE4){
//                 point_idx = sorted_point_idx_list[right_index];
//                 neural_points.get_xyz_data(point_idx, pc_xyz);
//                 dist = distance_between(sp_xyz, pc_xyz);
//                 if (dist < options.radius_threshold){
//                     tmp_point_buffer[tmp_point_buffer_size] = PixSpec(dist, point_idx);
//                     tmp_point_buffer_size++;
//                     if (tmp_point_buffer_size >= POINT_BUFFER_SIZE4) break;
//                 }
//                 else{
//                     break;
//                 }
//                 right_index++;
//             }
//         }
//         __syncwarp((1U << kernel_size2) - 1);
//         int start_index = atomicAdd(&points_buffer_size[ray_blk_idx], tmp_point_buffer_size);
//         for (int i = 0; i < tmp_point_buffer_size; i++){
//             if (start_index + i >= POINT_BUFFER_SIZE48){
//                 points_buffer_size[ray_blk_idx] = POINT_BUFFER_SIZE48;
//                 break;
//             }
//             points_buffer[ray_blk_idx][start_index + i] = tmp_point_buffer[i];
//         }
//         // Step3.2: find the topK points for each ray
//         __syncwarp((1U << kernel_size2) - 1);
//         // if no points in the buffer, skip the following steps
//         if (points_buffer_size[ray_blk_idx] > 0){
//             if (lane_idx == 0){
//                 bubble_sort<PixSpec>(points_buffer[ray_blk_idx], points_buffer_size[ray_blk_idx], false);
//             }
//             __syncwarp((1U << kernel_size2) - 1);
//             // Step3.3: aggregate feature
//             // select topK points from sorted points_buffer
//             int valid_pc_num = min(options.topK, points_buffer_size[ray_blk_idx]);
//             float sigma = 0.0f;
//             float sp_sh[SH_SIZE] = {0.0f};
//             // intermediate gradient buffer
//             float weight = 0.0f;
//             int grad_point_idx = -1;
//             int grad_weight = 0.0f;

//             if (lane_idx < valid_pc_num){
//                 float sdf = points_buffer[ray_blk_idx][lane_idx].dist;
//                 int point_idx = points_buffer[ray_blk_idx][lane_idx].idx;
//                 weight = sdf2weight(sdf, options);
//                 sigma = neural_points.get_density_data(point_idx);
//                 sigma = sigma * weight;
//                 neural_points.get_sh_data(point_idx, sp_sh);
//                 cuda_util::_fmulf_vector(sp_sh, weight, SH_SIZE);
//                 grad_point_idx = point_idx;
//                 grad_weight = weight;
//             }

//             sigma = cub::WarpReduce<float>(sigma_buffer[ray_blk_idx]).Sum(sigma, kernel_size2);
//             weight = cub::WarpReduce<float>(weight_buffer[ray_blk_idx]).Sum(weight, kernel_size2);
//             weight = weight + 1e-8f;
//             weight = __shfl_sync((1U << kernel_size2) - 1, weight, 0); // broadcast sigma to all threads (warp) for calculating the same log_transmit
//             sigma = fdividef(sigma, weight);
//             sigma = __shfl_sync((1U << kernel_size2) - 1, sigma, 0); // broadcast sigma to all threads (warp) for calculating the same log_transmit

//             #pragma unroll 
//             for (int i = 0; i < SH_SIZE; i++){
//                 sp_sh[i] = WarpReduce(sh_buffer[ray_blk_idx]).Sum(sp_sh[i], kernel_size2);
//                 sp_sh[i] = fdividef(sp_sh[i], weight);
//                 sp_sh[i] = __shfl_sync((1U << kernel_size2) - 1, sp_sh[i], 0);
//             }
//             // Step4: volume rendering 
//             const float pcnt = options.t_intvl * fmaxf(sigma, 0.0f); 
//             const float color_weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
//             log_transmit -= pcnt;
            
//             // gadient calculation
//             float color[3] = {0.0f, 0.0f, 0.0f};
//             float total_color[3] = {0.0f, 0.0f, 0.0f};
//             float color_in_01[3] = {0.0f, 0.0f, 0.0f};

//             #pragma unroll
//             for (int i = 0; i < neural_points.basis_dim; i++){
//                 color[0] += sp_sh[i + neural_points.basis_dim * 0] * sphfunc_val[ray_blk_idx][i];
//                 color[1] += sp_sh[i + neural_points.basis_dim * 1] * sphfunc_val[ray_blk_idx][i];
//                 color[2] += sp_sh[i + neural_points.basis_dim * 2] * sphfunc_val[ray_blk_idx][i];
//             }

//             for (int i = 0; i < 3; i++){
//                 color[i] += 0.5f;
//                 total_color[i] = fmaxf(color[i], 0.f);
//                 color_in_01[i] = total_color[i] == color[i];
//                 total_color[i] *= grad_out[i];
//             }
//             float total_color_sum = total_color[0] + total_color[1] + total_color[2];
//             float grad_common[3] = {0.0f, 0.0f, 0.0f};
//             float curr_grad_color[SH_SIZE] = {0.0f};
//             for (int i = 0; i < 3; i++){
//                 grad_common[i] = color_weight * color_in_01[i] * grad_out[i];
//                 for (int j = 0; j < neural_points.basis_dim; j++){
//                     curr_grad_color[i*neural_points.basis_dim + j] = sphfunc_val[ray_blk_idx][j] * grad_common[i];
//                 }
//             }
//             accum -= color_weight * total_color_sum;
//             float curr_grad_sigma = options.t_intvl * (total_color_sum * _EXP(log_transmit) - accum);
            
//             if (lane_idx < valid_pc_num){
//                 interpolate_topk_backward_kernel(grads.grad_sh_out, 
//                                                  grads.grad_sigma_out,
//                                                  grads.mask_out,
//                                                  grad_point_idx,
//                                                  grad_weight,
//                                                  weight,
//                                                  SH_SIZE,
//                                                  curr_grad_color,
//                                                  curr_grad_sigma
//                                                  );
//             }
            
//             if (_EXP(log_transmit) < options.stop_threshold) break;
//         }
//         t += options.t_intvl;
//         points_buffer_size[ray_blk_idx] = 0;
//     }
// }

// }//namespace device


// torch::Tensor volume_render_epcq_fused_topk(// Input
//                               NeuralPointsSpec& neural_points, 
//                               CameraSpec& cam,
//                               RenderOptions& options,
//                               torch::Tensor& sorted_point_idx_list,
//                               torch::Tensor& sorted_depth_list,
//                               torch::Tensor& hashtable,
//                               torch::Tensor& ray_index,
//                               torch::Tensor& rgb_gt,
//                               // Output
//                               NeuralPointsGrads& grads
//                               ){
//     /*
//     *   This function performs the volume rendering of the neural points 
//     * containing the forward and backward pass.
//     */                                                                
//     neural_points.check();
//     cam.check();
//     grads.check();
//     CHECK_INPUT(sorted_point_idx_list);
//     CHECK_INPUT(sorted_depth_list);
//     CHECK_INPUT(hashtable);
//     CHECK_INPUT(ray_index);
//     CHECK_INPUT(rgb_gt);
//     assert (ray_index.size(0) == rgb_gt.size(0));
//     if (hashtable.dim() == 2){
//         hashtable = hashtable.view({cam.height, cam.width, 2});
//     }
//     if (hashtable.dim() != 3 || hashtable.size(0) != cam.height || hashtable.size(1) != cam.width || hashtable.size(2) != 2){
//         printf("hashtable size: %d, %d, %d\n", static_cast<int>(hashtable.size(0)), static_cast<int>(hashtable.size(1)), static_cast<int>(hashtable.size(2)));
//         throw std::runtime_error("hashtable size error");
//     }
    

//     torch::TensorOptions floatOptions = torch::TensorOptions()
//                                                .dtype(torch::kFloat32)
//                                                .device(sorted_depth_list.device());

//     torch::Tensor rgb_out = torch::zeros({ray_index.size(0), 3}, floatOptions);     
//     const int num_elem = ray_index.size(0);
//     const int num_threads = TRACE_RAY_CUDA_THREADS;
//     const int num_blocks = CUDA_N_BLOCKS_NEEDED(num_elem * WARP_SIZE, num_threads);
//     device::render_image_backward_kernel_with_early_stop_topk<<<num_blocks, num_threads>>>(neural_points,
//                                                                                            cam,
//                                                                                            options,
//                                                                                            sorted_point_idx_list.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
//                                                                                            sorted_depth_list.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
//                                                                                            hashtable.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
//                                                                                            ray_index.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
//                                                                                            rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//                                                                                            rgb_gt.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//                                                                                            grads);
//     CUDA_CHECK_ERRORS;
//     return rgb_out;
// }