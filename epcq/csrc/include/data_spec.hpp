#pragma once 
#include <torch/extension.h>
#include "util.hpp"

#define PYBIND11_HIDDEN __attribute__((visibility("hidden")))
#define PYBIND11_DEFAULT __attribute__((visibility("default")))

struct PYBIND11_HIDDEN NeuralPointsSpec{
    torch::Tensor xyz_data;     // [N, 3]
    inline void check(){
        CHECK_INPUT(xyz_data);
    }
};

struct PYBIND11_HIDDEN CameraSpec{
    torch::Tensor c2w;
    torch::Tensor w2c;
    float fx;
    float fy;
    float cx;
    float cy;
    int width;
    int height;
    inline void check(){
        CHECK_INPUT(c2w);
        CHECK_INPUT(w2c);
        TORCH_CHECK(c2w.is_floating_point());
        TORCH_CHECK(c2w.ndimension() == 2);
        TORCH_CHECK(c2w.size(1) == 4);
    }
};

// enum RenderStrategies{
//     EPCQ_PER_THREAD_PER_SP,
//     EPCQ_PER_THREAD_PER_RAY,
//     EPCQ_PER_THREAD_PER_RAY_TOPK
// };

// enum SDF2WEIGHT_TYPE{
//     DISTANCE_INVERSE_WEIGHT,
//     GAUSSIAN_WEIGHT,
// };

// enum RasterizeStrategies{
//     COMPACT_ARRAY,
//     HASH_TABLE
// };

struct PYBIND11_HIDDEN RenderOptions{
    int kernel_size;
    float t_intvl;
    float radius_threshold;
    float sigma_threshold;
    float background_brightness;
    float stop_threshold;
    int num_sample_points_per_ray;
    int num_point_cloud_per_sp;
    int num_point_per_ray;
    
    // RasterizeStrategies rasterize_strategy;

    // RenderStrategies render_strategy;
    // // sdf2weight
    // SDF2WEIGHT_TYPE sdf2weight_type;
    float sdf2weight_gaussian_alpha;
    float sdf2weight_gaussian_gamma;

    int topK;
};


struct PYBIND11_HIDDEN NeuralPointsGrads{
    torch::Tensor grad_sigma_out;
    torch::Tensor grad_sh_out;
    torch::Tensor mask_out;
    inline void check(){
        CHECK_INPUT(grad_sigma_out);
        CHECK_INPUT(grad_sh_out);
        CHECK_INPUT(mask_out);
    };
};
