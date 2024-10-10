
#include <torch/extension.h>
#include "data_spec.hpp"

std::tuple<torch::Tensor, torch::Tensor> scatter(const torch::Tensor& , const torch::Tensor& ,const torch::Tensor& ,std::tuple<int, int> ,const int& );
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> scatter_hashtable(const torch::Tensor&,  const torch::Tensor&, const torch::Tensor&, std::tuple<int, int>, bool);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> quick_sampling(NeuralPointsSpec&, CameraSpec&, RenderOptions&, torch::Tensor&, torch::Tensor&);
torch::Tensor quick_query_for_nearby_point(NeuralPointsSpec&, CameraSpec&, RenderOptions&, torch::Tensor&, torch::Tensor&, torch::Tensor&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#define _REG_FUNC(funname) m.def(#funname, &funname)
    _REG_FUNC(scatter);
    _REG_FUNC(scatter_hashtable);
    _REG_FUNC(quick_sampling);
    _REG_FUNC(quick_query_for_nearby_point);


#undef _REG_FUNC

    py::class_<NeuralPointsSpec>(m, "NeuralPointsSpec")
        .def(py::init<>())
        .def_readwrite("xyz_data", &NeuralPointsSpec::xyz_data);
    
    py::class_<CameraSpec>(m, "CameraSpec")
        .def(py::init<>())
        .def_readwrite("c2w", &CameraSpec::c2w)
        .def_readwrite("w2c", &CameraSpec::w2c)
        .def_readwrite("fx", &CameraSpec::fx)
        .def_readwrite("fy", &CameraSpec::fy)
        .def_readwrite("cx", &CameraSpec::cx)
        .def_readwrite("cy", &CameraSpec::cy)
        .def_readwrite("width", &CameraSpec::width)
        .def_readwrite("height", &CameraSpec::height);

    py::class_<RenderOptions>(m, "RenderOptions")
        .def(py::init<>())
        .def_readwrite("kernel_size", &RenderOptions::kernel_size)
        .def_readwrite("t_intvl", &RenderOptions::t_intvl)
        .def_readwrite("radius_threshold", &RenderOptions::radius_threshold)
        .def_readwrite("sigma_threshold", &RenderOptions::sigma_threshold)
        .def_readwrite("background_brightness", &RenderOptions::background_brightness)
        .def_readwrite("stop_threshold", &RenderOptions::stop_threshold)
        .def_readwrite("num_sample_points_per_ray", &RenderOptions::num_sample_points_per_ray)
        .def_readwrite("num_point_cloud_per_sp", &RenderOptions::num_point_cloud_per_sp)
        .def_readwrite("num_point_per_ray", &RenderOptions::num_point_per_ray)
        .def_readwrite("sdf2weight_gaussian_alpha", &RenderOptions::sdf2weight_gaussian_alpha)
        .def_readwrite("sdf2weight_gaussian_gamma", &RenderOptions::sdf2weight_gaussian_gamma)
        .def_readwrite("topK", &RenderOptions::topK);


    py::class_<NeuralPointsGrads>(m, "NeuralPointsGrads")
        .def(py::init<>())
        .def_readwrite("grad_sigma_out", &NeuralPointsGrads::grad_sigma_out)
        .def_readwrite("grad_sh_out", &NeuralPointsGrads::grad_sh_out)
        .def_readwrite("mask_out", &NeuralPointsGrads::mask_out);
}

#ifdef VERBOSE
std::cout << "EPCQ C++ Extension Loaded!" << std::endl;
#endif