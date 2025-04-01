#include <pybind11/pybind11.h>  
#include <pybind11/stl.h> 
#include "renderer2D.hpp"

namespace py = pybind11;

class PyRenderer2D : public Renderer2D 
{
    public:
        using Renderer2D::Renderer2D;

        void UserDraw() override 
        {
            PYBIND11_OVERRIDE(void, Renderer2D, UserDraw);
        }
};

PYBIND11_MODULE(IzerRaster, m)
{
    m.doc() = "IzerRaster is a rasterizer.";

    py::class_<RGBA>(m, "RGBA")
        .def(py::init<uint8_t, uint8_t, uint8_t, uint8_t>())
        .def_readwrite("r", &RGBA::r)
        .def_readwrite("g", &RGBA::g)
        .def_readwrite("b", &RGBA::b)
        .def_readwrite("a", &RGBA::a)
        .def("__repr__",
                [](const RGBA &c) 
                {
                    return "<RGBA r=" + std::to_string(c.r) +
                           " g=" + std::to_string(c.g) +
                           " b=" + std::to_string(c.b) + 
                           " a=" + std::to_string(c.a) + ">";
                });

    py::class_<Renderer2D, PyRenderer2D>(m, "Renderer2D")
        .def(py::init<const std::string&, int, int>(),
                py::arg("appName") = "Renderer2D",
                py::arg("width") = 640,
                py::arg("height") = 480)
        .def("Init", &Renderer2D::Init)
        .def("Run", &Renderer2D::Run)
        .def("HandleEvents", &Renderer2D::HandleEvents)
         //.def("Render", &Renderer2D::Render)
        //.def("UserDraw", &Renderer2D::UserDraw)
        .def("Quit", &Renderer2D::Quit)
        .def("clearScreen", &Renderer2D::clearScreen)
        .def("drawPoint", py::overload_cast<int, int, uint32_t>(&Renderer2D::drawPoint), py::arg("x"), py::arg("y"), py::arg("rgba_uint32"))
        .def("drawPoint", py::overload_cast<int, int, RGBA>(&Renderer2D::drawPoint), py::arg("x"), py::arg("y"), py::arg("rgba_struct"))
        .def("drawLine", &Renderer2D::drawLine, py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"));
}
