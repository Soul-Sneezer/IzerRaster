#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <string>
#include <vector>

struct RGBA 
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;

    RGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a) : r(r), g(g), b(b), a(a) {}
};

class Renderer2D
{
private:
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* screenTexture = nullptr;
    std::vector<uint32_t> screenBuffer;
    
    TTF_Font* font; 
    uint32_t lastTime = SDL_GetTicks(); // Time since SDL_Init
    uint32_t frameCount = 0;            // Frames since last FPS update
    uint32_t fps = 0;                   // Current FPS value
    char fpsString[32] = "FPS: 0";
    std::string appName;
    bool isRunning;
    int windowWidth;
    int windowHeight;

public:
    Renderer2D(const std::string appName = "Renderer2D", int width = 640, int height = 480);
    
    void Init();

    void Run();
    void HandleEvents(); // WIP
    void Render();
    virtual void UserDraw(); // override this to draw
    void Quit();
    void clearScreen();
    void drawPoint(int x, int y, uint32_t rgba);
    void drawPoint(int x, int y, RGBA rgba);
    void drawLine(int x1, int y1, int x2, int y2);
};
