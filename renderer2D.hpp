#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <string>

class Renderer2D
{
private:
    SDL_Window* window;
    SDL_Renderer* renderer;
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
    void drawPoint(int x, int y);
    void drawLine(int x1, int y1, int x2, int y2);
};
