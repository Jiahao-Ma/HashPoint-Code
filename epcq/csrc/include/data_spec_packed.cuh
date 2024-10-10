#pragma once
#include <torch/extension.h>
#include "data_spec.hpp"

namespace device{
struct PackedNeuralPointsSpec{
    PackedNeuralPointsSpec(NeuralPointsSpec& spec)
    :
    xyz_data(spec.xyz_data.packed_accessor32<float, 2, torch::RestrictPtrTraits>())
    {}


    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> xyz_data;

    
    __device__ void get_xyz_data(const int& idx, float* data) const {
#pragma unroll 3
        for (int i = 0; i < 3; ++i){
            data[i] = this->xyz_data[idx][i];
        }
    }

};

struct PackedCameraSpec{
    PackedCameraSpec(CameraSpec& spec)
    :
    c2w(spec.c2w.packed_accessor32<float, 2, torch::RestrictPtrTraits>()),
    w2c(spec.w2c.packed_accessor32<float, 2, torch::RestrictPtrTraits>()),
    fx(spec.fx),
    fy(spec.fy),
    cx(spec.cx),
    cy(spec.cy),
    width(spec.width),
    height(spec.height) {}

    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> c2w;
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> w2c;
    float fx;
    float fy;
    float cx;
    float cy;
    int width;
    int height;  
};

struct PackedNeuralPointsGrads{
    PackedNeuralPointsGrads(NeuralPointsGrads& spec)
    :
    grad_sigma_out(spec.grad_sigma_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>()),
    grad_sh_out (spec.grad_sh_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>()),
    mask_out (spec.mask_out.packed_accessor32<bool, 1, torch::RestrictPtrTraits>())
    {}
    
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_sigma_out;
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> grad_sh_out;
    torch::PackedTensorAccessor32<bool, 1, torch::RestrictPtrTraits> mask_out;
};
} // namespace device
