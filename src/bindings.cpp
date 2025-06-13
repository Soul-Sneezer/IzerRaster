#define SDL_MAIN_HANDLED
#include <pybind11/pybind11.h>  
#include <pybind11/stl.h> 
#include "renderer2D.hpp"
#include <texture.hpp>


#ifdef HAS_CUDA
    #include <texture.hpp>
#endif


namespace py = pybind11;

class PyRenderer2D : public Renderer2D 
{
    public:
        using Renderer2D::Renderer2D;

      void UserUpdate() override {
        // Always call the Python override if it exists (handles input, transformations, etc.)
        PYBIND11_OVERRIDE(void, Renderer2D, UserUpdate);
        // Then call base class implementation to perform rendering (GPU or CPU drawing)
        
            Renderer2D::UserUpdate();  // run CUDA pipeline for GPU mode
    }
};

PYBIND11_MODULE(IzerRaster, m)
{
    m.doc() = "IzerRaster is a rasterizer.";
 
    m.attr("KEY_UNKNOWN") = SDLK_UNKNOWN;
    m.attr("KEY_RETURN") = SDLK_RETURN;
    m.attr("KEY_ESCAPE") = SDLK_ESCAPE;
    m.attr("KEY_BACKSPACE") = SDLK_BACKSPACE;
    m.attr("KEY_TAB") = SDLK_TAB;
    m.attr("KEY_SPACE") = SDLK_SPACE;
    m.attr("KEY_EXCLAIM") = SDLK_EXCLAIM;
    m.attr("KEY_DBLAPOSTROPHE") = SDLK_DBLAPOSTROPHE;
    m.attr("KEY_HASH") = SDLK_HASH;
    m.attr("KEY_DOLLAR") = SDLK_DOLLAR;
    m.attr("KEY_PERCENT") = SDLK_PERCENT;
    m.attr("KEY_AMPERSAND") = SDLK_AMPERSAND;
    m.attr("KEY_APOSTROPHE") = SDLK_APOSTROPHE;
    m.attr("KEY_LEFTPAREN") = SDLK_LEFTPAREN;
    m.attr("KEY_RIGHTPAREN") = SDLK_RIGHTPAREN;
    m.attr("KEY_ASTERISK") = SDLK_ASTERISK;
    m.attr("KEY_PLUS") = SDLK_PLUS;
    m.attr("KEY_COMMA") = SDLK_COMMA;
    m.attr("KEY_MINUS") = SDLK_MINUS;
    m.attr("KEY_PERIOD") = SDLK_PERIOD;
    m.attr("KEY_SLASH") = SDLK_SLASH;

    m.attr("KEY_LEFTBRACE") = SDLK_LEFTBRACE;
    m.attr("KEY_PIPE") = SDLK_PIPE;
    m.attr("KEY_RIGHTBRACE") = SDLK_RIGHTBRACE;
    m.attr("KEY_TILDE") = SDLK_TILDE;
    m.attr("KEY_DELETE") = SDLK_DELETE;
    m.attr("KEY_PLUSMINUS") = SDLK_PLUSMINUS;
    m.attr("KEY_CAPSLOCK") = SDLK_CAPSLOCK;

    m.attr("KEY_UP") = SDLK_UP;
    m.attr("KEY_DOWN") = SDLK_DOWN;
    m.attr("KEY_LEFT") = SDLK_LEFT;
    m.attr("KEY_RIGHT") = SDLK_RIGHT;

    m.attr("KEY_1") = SDLK_1;
    m.attr("KEY_2") = SDLK_2;
    m.attr("KEY_3") = SDLK_3;
    m.attr("KEY_4") = SDLK_4;
    m.attr("KEY_5") = SDLK_5;
    m.attr("KEY_6") = SDLK_6;
    m.attr("KEY_7") = SDLK_7;
    m.attr("KEY_8") = SDLK_8;
    m.attr("KEY_9") = SDLK_9;
    m.attr("KEY_0") = SDLK_0;

    m.attr("KEY_A") = SDLK_A;
    m.attr("KEY_B") = SDLK_B;
    m.attr("KEY_C") = SDLK_C;
    m.attr("KEY_D") = SDLK_D;
    m.attr("KEY_E") = SDLK_E;
    m.attr("KEY_F") = SDLK_F;
    m.attr("KEY_G") = SDLK_G;
    m.attr("KEY_H") = SDLK_H;
    m.attr("KEY_I") = SDLK_I;
    m.attr("KEY_J") = SDLK_J;
    m.attr("KEY_K") = SDLK_K;
    m.attr("KEY_L") = SDLK_L;
    m.attr("KEY_M") = SDLK_M;
    m.attr("KEY_N") = SDLK_N;
    m.attr("KEY_O") = SDLK_O;
    m.attr("KEY_P") = SDLK_P;
    m.attr("KEY_Q") = SDLK_Q;
    m.attr("KEY_R") = SDLK_R;
    m.attr("KEY_S") = SDLK_S;
    m.attr("KEY_T") = SDLK_T;
    m.attr("KEY_U") = SDLK_U;
    m.attr("KEY_V") = SDLK_V;
    m.attr("KEY_W") = SDLK_W;
    m.attr("KEY_X") = SDLK_X;
    m.attr("KEY_Y") = SDLK_Y;
    m.attr("KEY_Z") = SDLK_Z;

    
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

    m.def("inverse", [](const glm::mat4 &m)
    {
        return glm::inverse(m);
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

    py::class_<InputEvent>(m, "InputEvent")
        .def_readonly("type", &InputEvent::type)
        .def_readonly("key", &InputEvent::key)
        .def_readonly("mouseX", &InputEvent::mouseX)
        .def_readonly("mouseY", &InputEvent::mouseY)
        .def_readonly("wheelY", &InputEvent::wheelY) 
        .def_readonly("wheelX", &InputEvent::wheelX);

    py::class_<mesh>(m, "Mesh")
           .def("LoadFromObjectFile", &mesh::LoadFromObjectFile);


        /* ==================== Texture ====================== */
/*  Py nu va elibera obiectul (deținut de Renderer2D)  */
py::class_<Texture, std::unique_ptr<Texture, py::nodelete>>(m, "Texture")
    .def_property_readonly("width",  [](const Texture &t) { return t.w; })
    .def_property_readonly("height", [](const Texture &t) { return t.h; });

py::enum_<RenderMode>(m, "RenderMode")
        .value("WIREFRAME", RenderMode::WIREFRAME)
        .value("SHADED",    RenderMode::SHADED)
        .value("SHADED_WIREFRAME", RenderMode::SHADED_WIREFRAME)
        .value("TEXTURED", RenderMode::TEXTURED)
        .value("TEXTURED_WIREFRAME", RenderMode::TEXTURED_WIREFRAME);

    py::class_<Renderer2D, PyRenderer2D>(m, "Renderer2D")
        .def(py::init<const std::string&, uint16_t, uint16_t>(),
             py::arg("appName") = "Renderer2D",
             py::arg("width") = 640,
             py::arg("height") = 480)
        .def("Init", &Renderer2D::Init)
        .def("Run", &Renderer2D::Run, py::call_guard<py::gil_scoped_release>())
        .def("Quit", &Renderer2D::Quit)
        .def("clearScreen", &Renderer2D::clearScreen)
        .def("drawPoint", py::overload_cast<uint16_t, uint16_t, RGBA>(&Renderer2D::drawPoint),
             py::arg("x"), py::arg("y"), py::arg("rgba_struct"))
        .def("drawLine", &Renderer2D::drawLine)
        .def("drawRect", &Renderer2D::drawRect)
        .def("fillRect", &Renderer2D::fillRect)
        .def("drawCircle", &Renderer2D::drawCircle)
        .def("fillCircle", &Renderer2D::fillCircle)
        .def("drawTriangle", &Renderer2D::drawTriangle)
        .def("fillTriangle", &Renderer2D::fillTriangle)
        .def("drawCube", &Renderer2D::drawCube)
        .def("loadObj", &Renderer2D::loadObj, py::arg("filename"))
        .def("loadStl", &Renderer2D::loadStl, py::arg("filename"))
        .def("drawObj", &Renderer2D::drawObj)
        .def("getCurrentTime", &Renderer2D::GetCurrentTime)
        .def("getDeltaTime", &Renderer2D::GetDeltaTime)
        .def("applyRenderMatrix", &Renderer2D::applyRenderMatrix)
        .def("poolInputEvents", &Renderer2D::poolInputEvents)
        .def("detectInputEvent", &Renderer2D::detectInputEvent)
        .def_readwrite("renderMode", &Renderer2D::mode);

#ifdef HAS_CUDA
    py::class_<Renderer2D, PyRenderer2D>(m "Renderer2D")
        .def("loadTexture", &Renderer2D::loadTexture,py::return_value_policy::reference)   // NU transferă ownership
        .def("setCUDA", &Renderer2D::setCUDA)  
        .def("setTexture",  &Renderer2D::setTexture);
#endif
}
