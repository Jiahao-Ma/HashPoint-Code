// #include <torch/extension.h>
// #include "util.hpp"
// const int RMSPROP_STEP_CUDA_THREADS = 256;

// namespace device{
//     template<class T>
//     __host__ __device__ __inline__ T lerp(T a, T b, T w) {
//         return fmaf(w, b - a, a);
//     }
    
//     __device__ __inline__  void rmsprop_once(
//             float* __restrict__ ptr_data,
//             float* __restrict__ ptr_rms,
//             float* __restrict__ ptr_grad,
//             const float beta, const float lr, const float epsilon, float minval) {
//         float rms = *ptr_rms;
//         rms = rms == 0.f ? _SQR(*ptr_grad) : lerp(_SQR(*ptr_grad), rms, beta);
//         *ptr_rms = rms;
//         *ptr_data = fmaxf(*ptr_data - lr * (*ptr_grad) / (sqrtf(rms) + epsilon), minval);
//         // *ptr_grad = 0.f; // manually set to zero outside
//     }
//     __device__ __inline__ void rmsprop_once_pytorch(
//             float* __restrict__ ptr_data,
//             float* __restrict__ ptr_rms,
//             float* __restrict__ ptr_grad,
//             const float beta, const float lr, const float epsilon, float minval){
//         /*
//         // simple pytorch version of rmsprop
//         // Reference: https://pytorch.org/docs/stable/_modules/torch/optim/rmsprop.html#RMSprop
//         // No centered, differentiable and momentum here.
//         */
//         float rms = * ptr_rms;
//         rms = rms * beta + _SQR(*ptr_grad) * (1.f - beta); // no the case of `rms == 0.f`
//         *ptr_rms = rms;
//         *ptr_data = *ptr_data - lr * (*ptr_grad) / (sqrtf(rms) + epsilon); // no minval here
//         // *ptr_grad = 0.f; // manually set to zero outside
//     }
//     __global__ void rmsprop_mask_step_kernel(
//         torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> all_data,
//         torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> all_rms,
//         torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> all_grad,
//         const bool* __restrict__ mask,
//         float beta,
//         float lr,
//         float epsilon,
//         float minval,
//         float lr_last,
//         bool pytorch_mode = true
//         ){

//         CUDA_GET_THREAD_ID(tid, all_data.size(0) * all_data.size(1));
//         if (mask[tid / all_data.size(1)] == false) return;
//         int32_t chnl = tid % all_data.size(1);
//         if (pytorch_mode){
//             rmsprop_once_pytorch(all_data.data() + tid,
//                         all_rms.data() + tid,
//                         all_grad.data() + tid,
//                         beta,
//                         (chnl == all_data.size(1) - 1) ? lr_last : lr,
//                         epsilon,
//                         minval);

//         }
//         else{
//             rmsprop_once(all_data.data() + tid,
//                         all_rms.data() + tid,
//                         all_grad.data() + tid,
//                         beta,
//                         (chnl == all_data.size(1) - 1) ? lr_last : lr,
//                         epsilon,
//                         minval);
//         }
//     }

//     __device__ __inline__ void sgd_once(
//             float* __restrict__ ptr_data,
//             float* __restrict__ ptr_grad,
//             const float lr) {
//         *ptr_data -= lr * (*ptr_grad);
//         *ptr_grad = 0.f;
//     }

//     __global__ void sgd_mask_step_kernel(
//             torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> all_data,
//             torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> all_grad,
//             const bool* __restrict__ mask,
//             float lr,
//             float lr_last) {
//         CUDA_GET_THREAD_ID(tid, all_data.size(0) * all_data.size(1));
//         if (mask[tid / all_data.size(1)] == false) return;
//         int32_t chnl = tid % all_data.size(1);
//         sgd_once(all_data.data() + tid,
//                 all_grad.data() + tid,
//                 (chnl == all_data.size(1) - 1) ? lr_last : lr);
//     }

// }

// void rmsprop_step(
//         torch::Tensor data,
//         torch::Tensor rms,
//         torch::Tensor grad,
//         torch::Tensor indexer,
//         float beta,
//         float lr,
//         float epsilon,
//         float minval,
//         float lr_last,
//         bool pytorch_model
//         ) {
//     CHECK_INPUT(data);
//     CHECK_INPUT(rms);
//     CHECK_INPUT(grad);
//     CHECK_INPUT(indexer);

//     if (lr_last < 0.f) lr_last = lr;
//     const int cuda_n_threads = RMSPROP_STEP_CUDA_THREADS;
//     const size_t Q = data.size(0) * data.size(1);
//     const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
//     device::rmsprop_mask_step_kernel<<<blocks, cuda_n_threads>>>(
//             data.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//             rms.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//             grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//             indexer.data_ptr<bool>(),
//             beta,
//             lr,
//             epsilon,
//             minval,
//             lr_last,
//             pytorch_model);
//     CUDA_CHECK_ERRORS;
// }

// void sgd_step(
//         torch::Tensor data,
//         torch::Tensor grad,
//         torch::Tensor indexer,
//         float lr,
//         float lr_last) {

//     CHECK_INPUT(data);
//     CHECK_INPUT(grad);
//     CHECK_INPUT(indexer);

//     if (lr_last < 0.f) lr_last = lr;

//     const int cuda_n_threads = RMSPROP_STEP_CUDA_THREADS;
//     const size_t Q = data.size(0) * data.size(1);
//     const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
//     device::sgd_mask_step_kernel<<<blocks, cuda_n_threads>>>(
//             data.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//             grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
//             indexer.data_ptr<bool>(),
//             lr,
//             lr_last);
//     CUDA_CHECK_ERRORS;

// }