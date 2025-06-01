#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_events.h> 
#include <SDL3/SDL_keycode.h> 
#include <SDL3_ttf/SDL_ttf.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept> 
#include <iostream> 
#include <optional>
#include "texture.hpp"


struct RGBA 
{
    uint8_t b; // it has to be bgr, instead of rgb, or the colours will be reversed when actually drawing to screen
               // so RGBA(0, 0, 255, 255) would actually be red, if uint8_t r was first
    uint8_t g;
    uint8_t r;
    uint8_t a;

    RGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a) : r(r), g(g), b(b), a(a) {}
    RGBA() : r(0), g(0), b(0), a(255) {}
};

struct triangle {
    glm::vec4 p[3];   // poziţii
    glm::vec2 t[3];   // **NEW** coordonate UV
};




struct mesh 
{
    std::vector<triangle> tris;
    Texture*  texture = nullptr;

    bool LoadFromObjectFile(const std::string& sFilename);
};

struct InputEvent 
{
    std::string type;
    int key;
    int mouseX;
    int mouseY;
    int wheelY;
    int wheelX;
};

enum class RenderMode {
    WIREFRAME,              // doar muchii
    SHADED,                 // culori plane
    SHADED_WIREFRAME,       // culori + sârmă
    TEXTURED,               // textură (fără sârmă)
    TEXTURED_WIREFRAME      // textură + sârmă
};
struct Tri {
    glm::vec2 p[3];   // screen-space xy
    float     z[3];   // adâncimi
    glm::vec2 t[3];   //  ← ADĂUGĂ UV-urile!
    float     depth;
};

class Renderer2D
{
private:
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* screenTexture = nullptr;
    std::vector<RGBA> screenBuffer;
    Texture* currentTex = nullptr;   // pointer la textura activă
    mesh*        currentMesh = nullptr;  

    //stuff for projections
    mesh meshCube;
    mesh obj;
    glm::mat4 rotX, rotZ, transl, proj;
    float theta;
    glm::vec4 cameraPos;

       
    TTF_Font* font; 
    uint32_t frameCount = 0;            // Frames since last FPS update
    uint32_t fps = 0;                   // Current FPS value
    char fpsString[32] = "FPS: 0";
    std::string appName;
    bool isRunning;
    int windowWidth;
    int windowHeight;

std::vector<float> depthBufferCPU;   //  ← NOU!
              // deja există

    float    translate       = 8.0f;
    float    pitch_angle     = 0.0f;
    bool     free_rotate     = false;
    float    free_theta      = 0.0f;
    float    rotation_step   = 0.1f;

    Uint64 perfFreq = 0;
    Uint64 lastPerfCounter = 0;


    static uint64_t lastTime;


    void drawHorizontalLine(uint16_t y, uint16_t x1, uint16_t x2, RGBA rgba);
    void update(float deltaTime);
    void simpleRender(mesh meshObj);
    void fillBottomFlatTriangle(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t x3, uint16_t y3, RGBA rgba);
    void fillTopFlatTriangle(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t x3, uint16_t y3, RGBA rgba);

protected:
    bool useGPU;
    uint32_t* cudaPixelBuffer = nullptr;
    float* cudaDepthBuffer = nullptr;
     glm::mat4 currentTransform {1.0f};

public:
    RenderMode mode;

    Renderer2D(const std::string appName = "Renderer2D", uint16_t width = 640, uint16_t height = 480);
    
    mesh applyRenderMatrix(glm::mat4 mat, mesh objMesh);
    void Init();
    uint64_t GetCurrentTime();
    float GetDeltaTime();
    void Run();
    void Render();
    virtual void UserUpdate(); // override this to draw
    void Quit();
    void clearScreen();
    void drawPoint(uint16_t x, uint16_t y, RGBA rgba);
    void drawLine(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGBA rgba);
    void drawRect(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGBA rgba);
    void fillRect(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGBA rgba);
    void drawCircle(uint16_t x, uint16_t y, uint16_t radius, RGBA rgba);
    void fillCircle(uint16_t x, uint16_t y, uint16_t radius, RGBA rgba);
    void drawTriangle(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t x3, uint16_t y3, RGBA rgba);
    void fillTriangle(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t x3, uint16_t y3, RGBA rgba);
    void drawCube();
    void gpuRender(const mesh& meshObj);
    mesh loadObj(std::string path);
    void drawObj(mesh obj);
    std::vector<InputEvent> poolInputEvents();
    std::optional<InputEvent> detectInputEvent();
    Texture* loadTexture(const std::string& path);
    void     setTexture(Texture* t);
    void fillTexturedTri(const triangle& tri, const Texture* tex);
};
