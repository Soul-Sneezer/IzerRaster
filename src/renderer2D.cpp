#include "renderer2D.hpp"
#include <iostream>
#include <SDL3_ttf/SDL_ttf.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

    Renderer2D::Renderer2D(const std::string appName, int width, int height) : appName(appName), windowWidth(width), windowHeight(height)
    {

    }

    void Renderer2D::Init()
    {
        SDL_SetAppMetadata(appName.c_str(), "1.0", "renderer");

        if (!SDL_Init(SDL_INIT_VIDEO)) {
            std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
            return ;
        }

        /*
        if (!TTF_Init())
        {
            std::cerr << "SDL_ttf could not initialize!"<< std::endl;
            SDL_Quit();
            return;
        }
        */

        if (!SDL_CreateWindowAndRenderer(appName.c_str(), windowWidth, windowHeight, 0, &window, &renderer)) {
            std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
            //TTF_Quit();
            SDL_Quit();

            return;
        }
        
        this->screenTexture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, windowWidth, windowHeight);
        if (!screenTexture) {
            std::cerr << "Screen texture could not be created! SDL Error: " << SDL_GetError() << std::endl;
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
            //TTF_Quit();
            SDL_Quit();
            
            return;
        }

        this->screenBuffer.resize(windowWidth * windowHeight);

        //font = TTF_OpenFont("arial.ttf", 24);

        if (!font)
        {
            return;
        }
        isRunning = true;
    }

    void Renderer2D::Run()
    {
        while (isRunning)
        {
            HandleEvents();

            Render();
        }
    }

    void Renderer2D::HandleEvents()
    {
        SDL_Event event;

        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
                case SDL_EVENT_QUIT:
                    isRunning  = false;
                    break;
                default:
                    break;
            }
        }
    }

    void Renderer2D::Render()
    {
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
        SDL_RenderClear(renderer);

        UserDraw();
        
        SDL_UpdateTexture(screenTexture, nullptr, screenBuffer.data(),  windowWidth);

        SDL_RenderTexture(renderer, screenTexture, nullptr, nullptr);

        SDL_RenderPresent(renderer);
    }

    void Renderer2D::UserDraw()
    {
        //SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
        //drawLine(0, 0, windowWidth, windowHeight);
        
        drawPoint(300, 400, {255, 255, 255, 255});
        drawPoint(250, 320, {100, 120, 180, 255});
        drawPoint(500, 200, {200, 50, 40, 200});
    }

    void Renderer2D::Quit()
    {
        if (!isRunning && window == nullptr && renderer == nullptr) {
            return;
        }

        if (renderer) {
            SDL_DestroyRenderer(renderer);
            renderer = nullptr;
        }
        if (window) {
            SDL_DestroyWindow(window);
            window = nullptr;
        }

        // TTF_CloseFont(font);
        // TTF_Quit();

        SDL_Quit(); 

        isRunning = false;
    }

    void Renderer2D::clearScreen()
    {
        SDL_RenderClear(renderer);
    }

    void Renderer2D::drawPoint(int x, int y, uint32_t rgba)
    {
        SDL_SetRenderDrawColor(renderer, (rgba << 24) & 0xFF, (rgba << 16) * 0xFF, (rgba << 8) * 0xFF, rgba * 0xFF);
        SDL_RenderPoint(renderer, (float)x, (float)y);
    }

    void Renderer2D::drawPoint(int x, int y, RGBA rgba)
    {
        SDL_SetRenderDrawColor(renderer, rgba.r, rgba.g, rgba.b, rgba.a);
        SDL_RenderPoint(renderer, (float)x, (float)y);

    }

    void Renderer2D::drawLine(int x1, int y1, int x2, int y2)
    {
        //SDL_RenderLine(renderer, (float)x1, (float)y1, (float)x2, (float)y2);
    }
