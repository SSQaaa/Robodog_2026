#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "orbbec_camera.cpp"

namespace py = pybind11;

PYBIND11_MODULE(orbbec_native, m) {
    py::class_<OrbbecCamera>(m, "OrbbecCamera")
        .def(py::init<>())
        .def("start", &OrbbecCamera::start)
        .def("stop", &OrbbecCamera::stop)
        .def("get_color_frame", &OrbbecCamera::get_color_frame)
        .def("get_color_size", &OrbbecCamera::get_color_size)
        .def("get_depth_size", &OrbbecCamera::get_depth_size)
        .def("get_color_intrinsics", &OrbbecCamera::get_color_intrinsics)
        .def("get_depth_intrinsics", &OrbbecCamera::get_depth_intrinsics)
        .def("is_rotate_180_enabled", &OrbbecCamera::is_rotate_180_enabled)
        .def("get_depth", &OrbbecCamera::get_depth)
        .def("get_depth_in_box", &OrbbecCamera::get_depth_in_box);
}
