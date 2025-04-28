#define SDL_MAIN_HANDLED
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

        void UserInit() override 
        {
            PYBIND11_OVERRIDE(void, Renderer2D, UserInit);
        }

        void HandleEvents() override 
        {
            PYBIND11_OVERRIDE(void, Renderer2D, HandleEvents);
        }
};

PYBIND11_MODULE(IzerRaster, m)
{
    m.doc() = "IzerRaster is a rasterizer.";

    py::class_<glm::vec3>(m, "vec3")
           .def(py::init<float, float, float>())
           .def_readwrite("x", &glm::vec3::x)
           .def_readwrite("y", &glm::vec3::y)
           .def_readwrite("z", &glm::vec3::z)
           .def("__repr__", [](const glm::vec3 &v) 
            {
                return "<vec3 (" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")>";       
            });

    py::class_<glm::vec4>(m, "vec4")
        .def(py::init<float, float, float, float>())
        .def_readwrite("x", &glm::vec4::x)
        .def_readwrite("y", &glm::vec4::y)
        .def_readwrite("z", &glm::vec4::z)
        .def_readwrite("w", &glm::vec4::w)
        .def("__repr__", [](const glm::vec4 &v) 
        {
            return "<vec4 (" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ", " + std::to_string(v.w) + ")>";     
        });

    py::class_<glm::mat4>(m, "mat4")
        .def(py::init<>())
        .def("__mul__", [](const glm::mat4 &a, const glm::mat4 &b)
        {
            return a * b;
        }, py::is_operator())
        .def("__mul__", [](const glm::mat4 &m, const glm::vec4 &v)
        {
            return m * v;
        }, py::is_operator())
        .def("__repr__", [](const glm::mat4 &m)
        {
            std::string out = "<mat4:\n";
            for (int i = 0; i < 4; i++)
            {
                out += "  (";
                for (int j = 0; j < 4; j++)
                    out += std::to_string(m[i][j])  + (j < 3 ? ", " : "");
                out += ")\n";
            }
            out += ">";
            return out;
        });

    m.def("translate", [](const glm::vec3 &v)
    {
        return glm::translate(glm::mat4(1.0f), v);
    });

    m.def("rotate", [](float angle_radians, const glm::vec3& axis)
    {
        return glm::rotate(glm::mat4(1.0f), angle_radians, axis);
    });

    m.def("scale", [](const glm::vec3 & v)
    {
        return glm::scale(glm::mat4(1.0f), v);
    });

    m.def("transform_point", [](const glm::mat4 &m, const glm::vec3 &v)
    {
        glm::vec4 temp = m * glm::vec4(v, 1.0f);
        return glm::vec3(temp);
    });

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

    py::class_<mesh>(m, "Mesh")
           .def("LoadFromObjectFile", &mesh::LoadFromObjectFile);

    py::class_<Renderer2D, PyRenderer2D>(m, "Renderer2D")
        .def(py::init<const std::string&, uint16_t, uint16_t>(),
                py::arg("appName") = "Renderer2D",
                py::arg("width") = 640,
                py::arg("height") = 480)
        .def("Init", &Renderer2D::Init)
        .def("Run", &Renderer2D::Run)
        .def("HandleEvents", &Renderer2D::HandleEvents)
        .def("Quit", &Renderer2D::Quit)
        .def("clearScreen", &Renderer2D::clearScreen)
        .def("drawPoint", py::overload_cast<uint16_t, uint16_t, RGBA>(&Renderer2D::drawPoint), py::arg("x"), py::arg("y"), py::arg("rgba_struct"))
        .def("drawLine", &Renderer2D::drawLine, py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"), py::arg("rgba"))
        .def("drawRect", &Renderer2D::drawRect)
        .def("fillRect", &Renderer2D::fillRect)
        .def("drawCircle", &Renderer2D::drawCircle)
        .def("fillCircle", &Renderer2D::fillCircle)
        .def("drawTriangle", &Renderer2D::drawTriangle)
        .def("fillTriangle", &Renderer2D::fillTriangle)
        .def("drawCube", &Renderer2D::drawCube)
        .def("loadObj", &Renderer2D::loadObj,py::arg("filename"))
        .def("drawObj", &Renderer2D::drawObj)
        .def("getCurrentTime", &Renderer2D::GetCurrentTime)
        .def("getDeltaTime", &Renderer2D::GetDeltaTime)
        .def("applyRenderMatrix", &Renderer2D::applyRenderMatrix);
}
