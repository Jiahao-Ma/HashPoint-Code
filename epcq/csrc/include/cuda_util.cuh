// Declarations of utility functions for CUDA kernels
#pragma once
#include <torch/extension.h>
#include <limits>
#include <thrust/tuple.h>
#include "data_spec.hpp"
#include "data_spec_packed.cuh"

#define clamp(x, x_min, x_max) min(max(x, x_min), x_max)
#define clampf(x, x_min, x_max) fminf(fmaxf(x, x_min), x_max)
const int WARP_SIZE = 32;
const int POINT_BUFFER_SIZE = 32;
const int POINT_BUFFER_SIZE8 = 8;
const int POINT_BUFFER_SIZE4 = 4;
const int POINT_BUFFER_SIZE2 = 2;
const int POINT_BUFFER_SIZE1 = 1;
const int POINT_BUFFER_SIZE48 = 48;
const int DEFAULT_KERNEL_SIZE = 25; // 5 x 5 kernel
const int MAX_SAMPLE_POINTS_NUM = 128;
const int TRACE_RAY_CUDA_THREADS = 128;
const int TRACE_RAY_CUDA_RAYS_PER_BLOCK = TRACE_RAY_CUDA_THREADS / WARP_SIZE;
const int SH_SIZE = 3 * 9; // deg 3 by default (1 + 3 + 5 = 9 * 3 = 27)

namespace cuda_util{

template<class T>
__device__ inline bool _equal(const T& a, const T& b){
    return a == b;
}

template<>
__device__ inline bool _equal<float>(const float& a, const float& b){
    return fabs(a - b) < 1e-6;
}

template<>
__device__ inline bool _equal<double>(const double& a, const double& b){
    return fabs(a - b) < 1e-9;
}

template<class T>
__device__ inline void _swap(T& a, T& b){
    T tmp = a;
    a = b;
    b = tmp;
}

template<class T>
__device__ inline void _subtract3d(const T* a, const T* b, T* c){
#pragma unroll 3
    for (int i = 0; i < 3; i++){
        c[i] = a[i] - b[i];
    }
}

template<class T>
__device__ inline T _dot3d(const T *a, const T *b){
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__ inline float _norm(const float* __restrict__ x){
    return norm3df(x[0], x[1], x[2]); // sqrt(x[0]^2 + x[1]^2 + x[2]^2)
}

__device__ inline void _normalize(float* __restrict__ x){
    // normalize 3D vector
    float reverse_norm_v = rnorm3df(x[0], x[1], x[2]); // 1 / sqrt(x[0]^2 + x[1]^2 + x[2]^2)
    x[0] *= reverse_norm_v;
    x[1] *= reverse_norm_v;
    x[2] *= reverse_norm_v;
}
__device__ inline void _normalize_vector(float* __restrict__ x, const int n){
    float _sum = 0.0f;
    for (int i = 0; i < n; i++){
        _sum += x[i];
    }
    for (int i = 0; i < n; i++){
        // x[i] /= _sum;
        x[i] = fdividef(x[i], _sum);
    }
}

__device__ inline void _fmulf_vector(float* __restrict__ x, const float& weight, const int num){
#pragma unroll 
    for (int i = 0; i < num; i++){
        x[i] *= weight;
    }
}

__device__ inline void _fdividef_vector(float* __restrict__ x, const float& weight, const int num){
#pragma unroll 
    for (int i = 0; i < num; i++){
        // x[i] *= weight;
        x[i] = fdividef(x[i], weight);
    }
}

__device__ inline void _faddf_vector(float* __restrict__ x, float* __restrict__ y, const int num){
#pragma unroll
    for (int i = 0; i < num; i++){
        x[i] += y[i];
    }
}

} // namespace cuda_util


namespace device{

struct PointsSpec{
    int point_idx=0;
    float pc_xyz[3]={0.0f, 0.0f, 0.0f};
    PointsSpec() = default;
    __device__ PointsSpec(int point_idx, float* __restrict__ pc_xyz)
        : point_idx(point_idx){
            this->pc_xyz[0] = pc_xyz[0];
            this->pc_xyz[1] = pc_xyz[1];
            this->pc_xyz[2] = pc_xyz[2];
        }
    __device__ inline void set(int point_idx, float* __restrict__ pc_xyz){
        this->point_idx = point_idx;
        this->pc_xyz[0] = pc_xyz[0];
        this->pc_xyz[1] = pc_xyz[1];
        this->pc_xyz[2] = pc_xyz[2];
    }

};

struct RPointsSpec{
    int point_idx=0;
    float pc_xyz[3]={0.0f, 0.0f, 0.0f};
    float t = 0.0f;
    float sigma = 0.0f;
    RPointsSpec() = default;
    __device__ RPointsSpec(int point_idx, float* __restrict__ pc_xyz, float t, float sigma)
        : point_idx(point_idx), t(t), sigma(sigma){
            this->pc_xyz[0] = pc_xyz[0];
            this->pc_xyz[1] = pc_xyz[1];
            this->pc_xyz[2] = pc_xyz[2];
        }
    __device__ inline void set(int point_idx, float* __restrict__ pc_xyz, float t, float sigma){
        this->point_idx = point_idx;
        this->pc_xyz[0] = pc_xyz[0];
        this->pc_xyz[1] = pc_xyz[1];
        this->pc_xyz[2] = pc_xyz[2];
        this->t = t;
        this->sigma = sigma;
    }

};

struct PixSpec{
    float dist;
    int idx;
    
    PixSpec() = default;
    __device__ PixSpec(float dist, int idx) : dist(dist), idx(idx) {};
    __device__ inline bool operator<(const PixSpec& other){
            return dist < other.dist;
    }
    __device__ inline bool operator>(const PixSpec& other){
            return dist > other.dist;
    }
    __device__ inline void init(){
        dist = 1e6f;
        idx = -1;
    }
    __device__ inline void set(float dist, int idx){
        this->dist = dist;
        this->idx = idx;
    }
};

struct UVSpec{
    int u;
    int v;
    UVSpec() = default;
    __device__ UVSpec(int u, int v) : u(u), v(v) {};
};

struct Projection{
    float proj_pt[3];
    float t;
    float dist_p2r; 
    Projection() = default;
    __device__ Projection(float* __restrict__ proj_pt, float& t, float& dist_p2r)
        : t(t), dist_p2r(dist_p2r){
            this->proj_pt[0] = proj_pt[0];
            this->proj_pt[1] = proj_pt[1];
            this->proj_pt[2] = proj_pt[2];
        }

    __device__ inline void set(float* __restrict__ proj_pt, float& t, float& dist_p2r){
        this->proj_pt[0] = proj_pt[0];
        this->proj_pt[1] = proj_pt[1];
        this->proj_pt[2] = proj_pt[2];
        this->t = t;
        this->dist_p2r = dist_p2r;
    }
};

struct SingleRaySpec{
    SingleRaySpec() = default;
    __device__ SingleRaySpec(const float* __restrict__ origin, const float* __restrict__ dir)
        : origin{origin[0], origin[1], origin[2]},
          dir{dir[0], dir[1], dir[2]} {
                cuda_util::_normalize(this->dir); 
    }

    float origin[3] = {0.0f, 0.0f, 0.0f};
    float dir[3] = {0.0f, 0.0f, 0.0f}; // normalized direction by default
    float tmin=0.f;
    float tmax=0.f;

    __device__ inline void PointRayInterset(const float* __restrict__ point, Projection& proj){
        /*
            calculate the projection of point on ray, 
            return the projection point, t, and distance
                           *
                           | dist_p2r
        origin <-----------*------------ dir
               |<----t---->*proj_pt
            
        */ 
        float sub_p_o[3];
        cuda_util::_subtract3d<float>(point, origin, sub_p_o);
        float t = cuda_util::_dot3d<float>(sub_p_o, dir) / cuda_util::_dot3d<float>(dir, dir);
        float proj_pt[3];
        proj_pt[0] = origin[0] + t * dir[0];
        proj_pt[1] = origin[1] + t * dir[1];
        proj_pt[2] = origin[2] + t * dir[2];
        float dist_vector[3];
        cuda_util::_subtract3d<float>(point, proj_pt, dist_vector);
        float dist_p2r = cuda_util::_norm(dist_vector);
        proj.set(proj_pt, t, dist_p2r);
    }

    __device__ inline void PointRayInterset_4t(const float* __restrict__ point, float& t) const {
        /*
            calculate the projection of point on ray only for t (_4t)
                           *
                           | dist_p2r
        origin <-----------*------------ dir
               |<----t---->*proj_pt
            
        */ 
        float sub_p_o[3];
        cuda_util::_subtract3d<float>(point, origin, sub_p_o);
        t = cuda_util::_dot3d<float>(sub_p_o, dir) / cuda_util::_dot3d<float>(dir, dir);
    }
    
    __device__ inline float PointRayInterset_4t(const float* __restrict__ point) const {
        /*
            calculate the projection of point on ray only for t (_4t)
                           *
                           | dist_p2r
        origin <-----------*------------ dir
               |<----t---->*proj_pt
            
        */ 
        float sub_p_o[3];
        cuda_util::_subtract3d<float>(point, origin, sub_p_o);
        return cuda_util::_dot3d<float>(sub_p_o, dir) / cuda_util::_dot3d<float>(dir, dir);
    }


    __device__ inline void set(const float* __restrict__ origin, const float* __restrict__ dir){
#pragma unroll 3
        for(int i = 0; i < 3; i++){
            this->origin[i] = origin[i];
            this->dir[i] = dir[i];
        }
        cuda_util::_normalize(this->dir);
    }

    __device__ inline void ray_tracing(float* point, const float t){
#pragma unroll 3
        for (int i = 0; i < 3; i++){
            point[i] = this->origin[i] + t * this->dir[i];
        }
    }

};

__device__ inline void pts_world2cam(const float* __restrict__ pts, const PackedCameraSpec& cam, float* __restrict__ pts_cam){
    pts_cam[0] = pts[0] * cam.w2c[0][0] + pts[1] * cam.w2c[0][1] + pts[2] * cam.w2c[0][2] + cam.w2c[0][3];
    pts_cam[1] = pts[0] * cam.w2c[1][0] + pts[1] * cam.w2c[1][1] + pts[2] * cam.w2c[1][2] + cam.w2c[1][3];
    pts_cam[2] = pts[0] * cam.w2c[2][0] + pts[1] * cam.w2c[2][1] + pts[2] * cam.w2c[2][2] + cam.w2c[2][3];
}

template<class T>
__device__ inline void bubble_sort(T *data, int* index, const int& size, bool descending=true){
        for (int i = 0; i < size; i++){
            bool swapped = false;
            for(int j = 0; j < size - i - 1; j++){
                if (descending){ // descending: large -> small
                    if (data[j] < data[j + 1]){
                        cuda_util::_swap<T>(data[j], data[j + 1]);
                        cuda_util::_swap<int>(index[j], index[j + 1]);
                        swapped = true;
                    }
                }
                else{
                    if (data[j] > data[j + 1]){
                        cuda_util::_swap<T>(data[j], data[j + 1]);
                        cuda_util::_swap<int>(index[j], index[j + 1]);
                        swapped = true;
                    }
                }
            }
            if (!swapped) break;
        }
}

template<class T>
__device__ inline void bubble_sort(T *data, const int& size, 
                                   bool descending=true){
        for (int i = 0; i < size; i++){
            bool swapped = false;
            for(int j = 0; j < size - i - 1; j++){
                if (descending){
                    if (data[j] < data[j + 1]){
                        cuda_util::_swap<T>(data[j], data[j + 1]);
                        swapped = true;
                    }
                }
                else{ // ascending
                    if (data[j] > data[j + 1]){
                        cuda_util::_swap<T>(data[j], data[j + 1]);
                        swapped = true;
                    }
                }
            }
            if (!swapped) break;
        }
    }

__device__ inline void cam2world_ray(int u, int v, 
                        const PackedCameraSpec& cam,
                        //Output
                        float* __restrict__ origin,
                        float* __restrict__ dir){
        // OpenCV camera coordinate
        float x = (u + 0.5 - cam.cx) / cam.fx;
        float y = (v + 0.5 - cam.cy) / cam.fy;
        float z = 1;
        dir[0] = cam.c2w[0][0] * x + cam.c2w[0][1] * y + cam.c2w[0][2] * z;
        dir[1] = cam.c2w[1][0] * x + cam.c2w[1][1] * y + cam.c2w[1][2] * z;
        dir[2] = cam.c2w[2][0] * x + cam.c2w[2][1] * y + cam.c2w[2][2] * z;
        origin[0] = cam.c2w[0][3], origin[1] = cam.c2w[1][3], origin[2] = cam.c2w[2][3];

        // normalize dir
        cuda_util::_normalize(dir); 
    }

__device__ inline float distance_between(float* __restrict__ p1, float* __restrict__ p2){
    float sub_p1_p2[3] = {0.0f, 0.0f, 0.0f};
    cuda_util::_subtract3d<float>(p1, p2, sub_p1_p2);
    return cuda_util::_norm(sub_p1_p2);
}

__device__ inline float fatomicMin(float *addr, float value){
    float old = *addr, assumed;
    if (old <= value) return old; 
    do{
        assumed = old;
        old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
    }while(old != assumed);
    return old;
}

__device__ inline float fatomicMax(float *addr, float value){
    float old = *addr, assumed;
    if (old >= value) return old;
    do{
        assumed = old;
        old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
    }while(old != assumed);
    return old;
}

__device__ inline void calculate_neighbor_uv(const int& ray_cnt_v, const int& ray_cnt_u, const int& kernel_size, const int& height, const int& width, const int& lane_idx, int& v_valid, int& u_valid){
    const int kernel_offset = ceil(kernel_size / 2.0f); // 5 -> 3
    int v_offset = ceil((float)(lane_idx+1) / kernel_size) - kernel_offset; 
    int u_offset = (lane_idx % kernel_size) - (kernel_offset-1);
    v_valid = clamp(ray_cnt_v + v_offset, 0, height - 1);
    u_valid = clamp(ray_cnt_u + u_offset, 0, width - 1);
}

__device__ inline void calculate_neighbor_uv_list(const int& ray_cnt_v, const int& ray_cnt_u, const int& kernel_size, const int& height, const int& width, UVSpec* uv_list){
    const int kernel_offset = ceil(kernel_size / 2.0f); // 5 -> 3
    for (int i = 0; i < kernel_size * kernel_size; i++){
        int v_offset = ceil((float)(i+1) / kernel_size) - kernel_offset; 
        int u_offset = (i % kernel_size) - (kernel_offset-1);
        int v_valid = clamp(ray_cnt_v + v_offset, 0, height - 1);
        int u_valid = clamp(ray_cnt_u + u_offset, 0, width - 1);
        uv_list[i] = UVSpec(u_valid, v_valid);
    }
}

// __device__ inline float sdf2weight(float sdf, RenderOptions options){
//         if (options.sdf2weight_type == SDF2WEIGHT_TYPE::DISTANCE_INVERSE_WEIGHT){
//             return fdividef(1.0f, abs(sdf + 1e-6f));
//         }
//         else if (options.sdf2weight_type == SDF2WEIGHT_TYPE::GAUSSIAN_WEIGHT){
//             return _EXP(- options.sdf2weight_gaussian_alpha * abs(sdf));
//         }
//         else{
//             // report error and return
//             printf("ERROR: sdf2weight_type not supported!\n");
//             return;
//         }
//     }

__device__ inline void preprocess_ray_kernel( // input
                                           const PackedNeuralPointsSpec& neural_points,
                                           const int& t_min_point_idx,
                                           const int& t_max_point_idx,
                                           const SingleRaySpec& ray,
                                           // output 
                                           float& tmin,
                                           float& tmax
    ){
        // tmin 
        if (t_min_point_idx >= 0){
            float pc_xyz[3] = {0.0f};
            neural_points.get_xyz_data(t_min_point_idx, pc_xyz);
            ray.PointRayInterset_4t(pc_xyz, tmin);
        }
        if (t_max_point_idx >= 0){
            float pc_xyz[3] = {0.0f};
            neural_points.get_xyz_data(t_max_point_idx, pc_xyz);
            ray.PointRayInterset_4t(pc_xyz, tmax);
        }
    }
__device__ inline void preprocess_ray_kernel( // input
                                           const PackedNeuralPointsSpec& neural_points,
                                           const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> buffer_point_index, 
                                           const int& v_valid,
                                           const int& u_valid,
                                           const SingleRaySpec& ray,
                                           // output 
                                           int& buffer_size,
                                           float& tmin,
                                           float& tmax
    ){  
        // tmin
        int point_idx = buffer_point_index[v_valid][u_valid][0];
        
        if (point_idx < 0) return;
        float pc_xyz[3] = {0.0f};
        neural_points.get_xyz_data(point_idx, pc_xyz);
        ray.PointRayInterset_4t(pc_xyz, tmin);
        buffer_size = 0;
        for (int i = 0; i < buffer_point_index.size(2); i++){
            point_idx = buffer_point_index[v_valid][u_valid][i];
            if (point_idx >=0 && i == buffer_point_index.size(2)-1){
                buffer_size = buffer_point_index.size(2);
            }
            if (point_idx < 0){
                buffer_size = i;
                break;
            }
        }
        point_idx = buffer_point_index[v_valid][u_valid][buffer_size-1];
        neural_points.get_xyz_data(point_idx, pc_xyz);
        ray.PointRayInterset_4t(pc_xyz, tmax);
    }

template <typename T>
__device__ inline void dump_min(T* data, T& min_value, const int size){
    for (int i = 0; i < size; i++){
        if (min_value > data[i]) min_value = data[i];
    }
}

template <typename T>
__device__ inline thrust::tuple<T, int> dump_argmin(T* data, const int size){
    T min_value = std::numeric_limits<T>::max();
    int min_value_index = 0;
    for (int i = 0; i < size; i++){
        
        if (min_value > data[i]) {
            min_value = data[i];
            min_value_index = i;
        }
    }
    return thrust::make_tuple(min_value, min_value_index);
}

template <typename T>
__device__ inline void dump_max(T* data, T& max_value, const int size){
    for (int i = 0; i < size; i++){
        if (max_value < data[i]) max_value = data[i];
    }
}
/*
iterative_search_index: find the index of target in data by iterating each element if target does not exist in the data,
            find the nearest index of target in data
binary_search_index: find the index of target in data if target does not exist in the data,
            find the nearest index of target in data
binary_search: find the index of target in data if target does not exist in the data,
            return -1
*/
__device__ inline int iterative_search_index(const PackedNeuralPointsSpec& neural_points,
                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>sorted_point_idx_list,
                                          const int& start_idx,
                                          const int& buffer_size,
                                          float* __restrict__ target_xyz
                                          ){
    float min_dist = 1e6f;
    int min_dist_idx = start_idx;
    for (int i = start_idx; i < start_idx + buffer_size; i++){
        float pc_xyz[3] = {0.0f};
        neural_points.get_xyz_data(sorted_point_idx_list[i], pc_xyz);
        float dist = distance_between(target_xyz, pc_xyz);
        if (dist < min_dist){
            min_dist = dist;
            min_dist_idx = i;
        }
    }
    return min_dist_idx; 
}
__device__ inline int binary_search_index(const PackedNeuralPointsSpec& neural_points,
                                          const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>sorted_point_idx_list,
                                          const SingleRaySpec& ray,
                                          const int& start_idx,
                                          const int& buffer_size,
                                          const float& target_t
                                         ){
    int left = start_idx;
    int right = start_idx + buffer_size - 1;
    float pc_xyz[3] = {0.0f};
    while( left <= right){
        int mid = (left + right) / 2;
        neural_points.get_xyz_data(sorted_point_idx_list[mid], pc_xyz);
        float t = ray.PointRayInterset_4t(pc_xyz);
        if (cuda_util::_equal<float>(t, target_t)){
            return mid;
        }
        else if (t < target_t){
            left = mid + 1;
        }
        else{
            right = mid - 1;
        }
    }
    if (right < start_idx){
        return left;
    }
    else if (left > start_idx + buffer_size - 1){
        return right;
    }
    return left;

}

__device__ inline int binary_search(
        const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> &data,
        const int &target
        ){
        int left = 0;
        int right = data.size(0) - 1;
        int result = -1;
        while(left <= right){
            int mid = (left + right) / 2;
            if (cuda_util::_equal<int>(data[mid], target)){
                result = mid;
                right = mid - 1;
            }
            else if (data[mid] < target){
                left = mid + 1;
            }
            else{
                right = mid - 1;
            }
        }
        return result;
}

} // namespace device
