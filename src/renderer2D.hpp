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

struct triangle
{
    glm::vec4 p[3];
};

struct mesh 
{
    std::vector<triangle> tris;

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

class Renderer2D
{
private:
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* screenTexture = nullptr;
    std::vector<RGBA> screenBuffer;

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

    static uint64_t lastTime;

    void drawHorizontalLine(uint16_t y, uint16_t x1, uint16_t x2, RGBA rgba);
    void update(float deltaTime);
    void simpleRender(mesh meshObj);
    void fillBottomFlatTriangle(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t x3, uint16_t y3, RGBA rgba);
    void fillTopFlatTriangle(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t x3, uint16_t y3, RGBA rgba);

public:
    Renderer2D(const std::string appName = "Renderer2D", uint16_t width = 640, uint16_t height = 480);
    
    mesh applyRenderMatrix(glm::mat4 mat, mesh objMesh);
    void Init();
    uint64_t GetCurrentTime();
    float GetDeltaTime();
    void Run();
    void Render();
    virtual void UserUpdate(); // override this to draw
    virtual void UserInit(); // override this to perform actions during initialization
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
    mesh loadObj(std::string path);
    void drawObj(mesh obj);
    std::vector<InputEvent> poolInputEvents();
    std::optional<InputEvent> detectInputEvent();
};
