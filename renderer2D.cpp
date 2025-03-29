#include "renderer2D.hpp"

    Renderer2D::Renderer2D(const std::string appName, int width, int height) : appName(appName), windowWidth(width), windowHeight(height)
    {
        this->window = nullptr;
        this->renderer = nullptr;
    }

    void Renderer2D::Init()
    {
        SDL_SetAppMetadata(appName.c_str(), "1.0", "renderer");

        if (!SDL_Init(SDL_INIT_VIDEO)) {
            return ;
        }

        if (!SDL_CreateWindowAndRenderer(appName.c_str(), windowWidth, windowHeight, 0, &window, &renderer)) {
            return ;
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

        SDL_RenderPresent(renderer);
    }

    void Renderer2D::UserDraw()
    {
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
        drawLine(0, 0, windowWidth, windowHeight);
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
        SDL_Quit(); 

        isRunning = false;
    }

    void Renderer2D::clearScreen()
    {
        SDL_RenderClear(renderer);
    }

    void Renderer2D::drawPoint(int x, int y)
    {
        SDL_RenderPoint(renderer, (float)x, (float)y);
    }

    void Renderer2D::drawLine(int x1, int y1, int x2, int y2)
    {
        SDL_RenderLine(renderer, (float)x1, (float)y1, (float)x2, (float)y2);
    }
