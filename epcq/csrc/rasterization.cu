#include <tuple>
#include <torch/extension.h>
#include "util.hpp"
#include "data_spec.hpp"
#include "cuda_util.cuh"
namespace device{
__global__ void scatter_hashtable_kernel(torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>point_idx_list,
                                         torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>point2img_idx_list,
                                         torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>depth_list,
                                         const int num_elem,
                                         int*  sorted_point_idx_list,
                                         float*  sorted_depth_list,
                                         int* hash_table,
                                         const bool sort = true
                                         ){
    const int ray_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_index >= num_elem) return;
    int index_in_point2img = binary_search(point2img_idx_list, ray_index);
    if (!cuda_util::_equal<int>(index_in_point2img, -1)){
        int bin_size = 0;
        int start_idx = index_in_point2img;
        int end_idx = index_in_point2img;
        while(cuda_util::_equal<int>(point2img_idx_list[end_idx], ray_index)){
            sorted_point_idx_list[end_idx] = point_idx_list[end_idx];
            sorted_depth_list[end_idx] = depth_list[end_idx];
            bin_size ++;
            end_idx++;
        }
        if (sort)
            bubble_sort<float>(sorted_depth_list + start_idx, sorted_point_idx_list + start_idx, bin_size, false);
        hash_table[ray_index*2 + 0] = start_idx;
        hash_table[ray_index*2 + 1] = bin_size;
    }                                        
}

__global__ void scatter_kernel(torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>point_idx_list,
                               torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>point2img_idx_list,
                               torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>depth_list,
                               torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>buffer_depth,
                               torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>buffer_point_index
){
    const int ray_index = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ PixSpec sharedMemory[];
    if (ray_index >= buffer_depth.size(0)) return;
    int index_in_point2img = binary_search(point2img_idx_list, ray_index);
    if (!cuda_util::_equal<int>(index_in_point2img, -1)){
        int q_size = 0;
        float q_max_z = -1000;
        int q_max_idx = -1;
        int B = buffer_depth.size(1);
        PixSpec* q_buffer = sharedMemory + threadIdx.x * B;
        float depth = -1;
        int point_idx = 0;
        while(cuda_util::_equal<int>(point2img_idx_list[index_in_point2img], ray_index)){
            depth = depth_list[index_in_point2img];
            if (depth < 0) continue; // invalid depth
            point_idx = point_idx_list[index_in_point2img];
            if ( q_size < B){
                q_buffer[q_size] = {depth, point_idx};
                if (depth > q_max_z){
                    q_max_z = depth;
                    q_max_idx = q_size;
                }
                q_size ++;
            }
            else if (depth < q_max_z){
                q_buffer[q_max_idx] = {depth, point_idx};
                q_max_z = depth;
                for (int i = 0; i < B; i++){
                    if (q_buffer[i].dist > q_max_z){
                        q_max_z = q_buffer[i].dist;
                        q_max_idx = i;
                    }
                }
            }
            index_in_point2img++;
        }
        if(q_size>0){
            bubble_sort<PixSpec>(q_buffer, q_size, false); 
            for (int i = 0; i < q_size; i++){
                buffer_depth[ray_index][i] = q_buffer[i].dist;
                buffer_point_index[ray_index][i] = q_buffer[i].idx;
            }
        }
    }
}
}// namespace device
std::tuple<torch::Tensor, torch::Tensor> scatter(const torch::Tensor& point_idx_list, 
                                                 const torch::Tensor& point2img_idx_list,
                                                 const torch::Tensor& depth_list,
                                                 std::tuple<int, int> img_size,
                                                 const int& bins
){  
    CHECK_INPUT(point_idx_list);
    CHECK_INPUT(point2img_idx_list);
    CHECK_INPUT(depth_list);

    torch::TensorOptions intOptions = torch::TensorOptions()
                                            .dtype(torch::kInt32)
                                            .device(point_idx_list.device())
                                            .requires_grad(false);
    torch::TensorOptions floatOptions = torch::TensorOptions()
                                            .dtype(torch::kFloat32)
                                            .device(point_idx_list.device())
                                            .requires_grad(false);

    torch::Tensor buffer_depth = torch::full({std::get<0>(img_size) * std::get<1>(img_size), bins}, -1, floatOptions);
    torch::Tensor buffer_point_index = torch::full({std::get<0>(img_size) * std::get<1>(img_size), bins}, -1, intOptions);

    cudaDeviceProp prop;
    int current_device = 0;
    cudaGetDevice(&current_device); 
    cudaGetDeviceProperties(&prop, 0);
    int sharedMemorySize = prop.sharedMemPerBlock;
    // My device's (RTX4090) shared memory size is 49152(48KB) per block
    int block_size = sharedMemorySize / (bins * sizeof(device::PixSpec) * 1.2); // 1.2 is the safety factor

    const dim3 block(block_size, 1, 1);
    const dim3 grid(CUDA_N_BLOCKS_NEEDED(buffer_depth.size(0), block.x), 1, 1);

    device::scatter_kernel<<<grid, block, block.x * bins * sizeof(device::PixSpec)>>>(point_idx_list.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                      point2img_idx_list.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                                      depth_list.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                                      // Output
                                                                      buffer_depth.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                                      buffer_point_index.packed_accessor32<int, 2, torch::RestrictPtrTraits>());
    return std::tuple<torch::Tensor, torch::Tensor>{ buffer_depth, buffer_point_index };                                                                  
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> scatter_hashtable(const torch::Tensor& point_idx_list, 
                                                 const torch::Tensor& point2img_idx_list,
                                                 const torch::Tensor& depth_list,
                                                 std::tuple<int, int> img_size,
                                                 const bool sort
                                                 
){  
    CHECK_INPUT(point_idx_list);
    CHECK_INPUT(point2img_idx_list);
    CHECK_INPUT(depth_list);

    torch::TensorOptions intOptions = torch::TensorOptions()
                                            .dtype(torch::kInt32)
                                            .device(point_idx_list.device())
                                            .requires_grad(false);
    torch::TensorOptions floatOptions = torch::TensorOptions()
                                            .dtype(torch::kFloat32)
                                            .device(point_idx_list.device())
                                            .requires_grad(false);

    torch::Tensor sorted_point_idx_list = torch::full({point_idx_list.size(0)}, -1, intOptions);
    torch::Tensor sorted_depth_list = torch::full({depth_list.size(0)}, -1, floatOptions);
    torch::Tensor hash_table = torch::full({std::get<0>(img_size) * std::get<1>(img_size), 2}, -1, intOptions);

    const int num_elem = std::get<0>(img_size) * std::get<1>(img_size);
    const dim3 block(TRACE_RAY_CUDA_THREADS);
    const dim3 grid(CUDA_N_BLOCKS_NEEDED(num_elem, block.x));

    device::scatter_hashtable_kernel<<<grid, block>>>(// Input
                                                      point_idx_list.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                      point2img_idx_list.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                                      depth_list.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                      num_elem,
                                                      // Output
                                                      sorted_point_idx_list.data_ptr<int>(),
                                                      sorted_depth_list.data_ptr<float>(),
                                                      hash_table.data_ptr<int>(),
                                                      sort
                                                      );

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>{ sorted_point_idx_list, sorted_depth_list, hash_table};                                                                  
}



