#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
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

struct vec3d{
    float x,y,z;
};


struct vec2d{
    float u,v;
};

struct mat4x4{
    float m[4][4] = {0};
};

struct triangle{
    glm::vec4 p[3];
};

struct mesh {
    std::vector<triangle> tris;

    bool LoadFromObjectFile(const std::string& sFilename);
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
    glm::mat4 matRotX, matRotZ, matTrans, matProj;
    float fTheta;
    glm::vec4 vCamera;

    
    TTF_Font* font; 
    uint32_t lastTime = SDL_GetTicks(); // Time since SDL_Init
    uint32_t frameCount = 0;            // Frames since last FPS update
    uint32_t fps = 0;                   // Current FPS value
    char fpsString[32] = "FPS: 0";
    std::string appName;
    bool isRunning;
    int windowWidth;
    int windowHeight;

    void drawHorizontalLine(int y, int x1, int x2, RGBA rgba);
public:
    Renderer2D(const std::string appName = "Renderer2D", int width = 640, int height = 480);
    
    void Init();

    void Run();
    void HandleEvents(); // WIP
    void Render();
    virtual void UserDraw(); // override this to draw
    virtual void UserInit();
    void Quit();
    void clearScreen();
    void update(float deltaTime);
    void simpleRender(mesh meshObj);
    void drawPoint(int x, int y, RGBA rgba);
    void drawLine(int x1, int y1, int x2, int y2, RGBA rgba);
    void drawRect(int x1, int y1, int x2, int y2, RGBA rgba);
    void fillRect(int x1, int y1, int x2, int y2, RGBA rgba);
    void drawCircle(int x, int y, int radius, RGBA rgba);
    void fillCircle(int x, int y, int radius, RGBA rgba);
    void drawTriangle(int x1, int y1, int x2, int y2, int x3, int y3, RGBA rgba);
    void fillBottomFlatTriangle(int x1, int y1, int x2, int y2, int x3, int y3, RGBA rgba);
    void fillTopFlatTriangle(int x1, int y1, int x2, int y2, int x3, int y3, RGBA rgba);
    void fillTriangle(int x1, int y1, int x2, int y2, int x3, int y3, RGBA rgba);
    void drawCube();
    void loadObj(std::string path);
    void drawObj();
    void multiplyMatrixVector(vec3d &i, glm::vec4 &o, mat4x4 &m); 
};
