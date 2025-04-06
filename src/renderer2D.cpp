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
        
        SDL_UpdateTexture(screenTexture, nullptr, screenBuffer.data(),  windowWidth * sizeof(RGBA));

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
        std::fill(screenBuffer.begin(), screenBuffer.end(), RGBA());
    }

    void Renderer2D::drawPoint(int x, int y, RGBA rgba)
    {
        if (y < 0 || y >= windowHeight || x < 0 || x >= windowWidth)
            return;

        this->screenBuffer[y * windowWidth + x] = rgba; 
    }

    void Renderer2D::drawLine(int x1, int y1, int x2, int y2, RGBA rgba) // Brensenham's Line Algorithm
    {
        int dx = std::abs(x2 - x1);
        int dy = - std::abs(y2 - y1);
        int sx = (x1 < x2) ? 1 : -1;
        int sy = (y1 < y2) ? 1 : -1;

        int err = dx + dy;

        int currentX = x1;
        int currentY = y1;

        while (true)
        {
            drawPoint(currentX, currentY, rgba);

            if (currentX == x2 && currentY == y2)
                break;

            if (2 * err >= dy)
            {
                if (currentX == x2)
                    break; 
                err += dy; 
                currentX += sx;
            }

            if (2 * err <= dx)
            {
                if (currentY == y2)
                    break;
                err += dx;
                currentY += sy;
            }
        }

    }

    void Renderer2D::drawRect(int x1, int y1, int x2, int y2, RGBA rgba)
    {
       drawLine(x1, y1, x2, y1, rgba);
       drawLine(x1, y1, x1, y2, rgba);
       drawLine(x2, y2, x1, y2, rgba);
       drawLine(x2, y2, x2, y1, rgba);
    }

    void Renderer2D::fillRect(int x1, int y1, int x2, int y2, RGBA rgba)
    {
        int startX = std::min(x1, x2);
        int startY = std::min(y1, y2);
        int endX = std::max(x1, x2);
        int endY = std::max(y1, y2);

        for(int y = startY; y < endY; y++)
        {
            drawHorizontalLine(y, startX, endX, rgba);
        }
    }


    // https://en.wikipedia.org/wiki/Midpoint_circle_algorithm 
    void Renderer2D::drawCircle(int x, int y, int radius, RGBA rgba)
    {
        if (radius < 0)
            return;

        if (radius == 0)
        {
            drawPoint(x, y, rgba);
            return;
        }
        
        // midpoint circle algorithm 
        int startX = 0;
        int startY = radius;

        int decision = 3 - 2 * radius;

        auto plotOctants = [&](int offsetX, int offsetY) // so you divide the space into 8 parts, and draw in each of them
        {
            drawPoint(x + offsetX, y + offsetY, rgba);
            drawPoint(x - offsetX, y + offsetY, rgba);
            drawPoint(x + offsetX, y - offsetY, rgba);
            drawPoint(x - offsetX, y - offsetY, rgba);
            drawPoint(x + offsetY, y + offsetX, rgba);
            drawPoint(x - offsetY, y + offsetX, rgba);
            drawPoint(x + offsetY, y - offsetX, rgba);
            drawPoint(x - offsetY, y - offsetX, rgba);
        };

        plotOctants(startX, startY); // point first points on the axes

        while (startX <= startY)
        {
            startX++; // you always move horizontally 

            if (decision > 0) // if true, you also need to move vertically
            {
                // midpoints was outside the circle
                startY--; // y-- because the upper left corner is (0,0), so going up means, decrementing

                decision = decision + 4 * (startX - startY) + 10; 
            }
            else // midpoint was on the circle or inside it 
                decision = decision + 4 * startX + 6;

            plotOctants(startX, startY);
        }
    }
    
    void Renderer2D::drawHorizontalLine(int y, int x1, int x2, RGBA rgba)
    {
        if (x1 > x2)
            std::swap(x1, x2);

        if (y < 0 || y >= windowHeight)
            return; 

        int startX = std::max(0, x1);
        int endX = std::min(windowWidth, x2 + 1);

        if (startX >= endX)
            return; 

        auto rowStart = screenBuffer.begin() + y * windowWidth + startX;
        std::fill(rowStart, rowStart + (endX - startX), rgba);
    }

    void Renderer2D::fillCircle(int x, int y, int radius, RGBA rgba)
    {
        if (radius < 0)
            return;

        if (radius == 0)
        {
            drawPoint(x, y, rgba);
            return;
        }
        
        // midpoint circle algorithm 
        int startX = 0;
        int startY = radius;

        int decision = 3 - 2 * radius;

        auto plotHorizontalLines = [&](int offsetX, int offsetY) // this is the only thing that changes from the drawCircle function
        {
            drawHorizontalLine(y + offsetY, x - offsetX, x + offsetX, rgba); // bottom

            drawHorizontalLine(y - offsetY, x - offsetX, x + offsetX, rgba); // top 

            drawHorizontalLine(y + offsetX, x - offsetY, x + offsetY, rgba); // mid bottom 

            drawHorizontalLine(y - offsetX, x - offsetY, x + offsetY, rgba); // mid top
        };

        plotHorizontalLines(startX, startY); // point first points on the axes

        while (startX <= startY)
        {
            startX++; // you always move horizontally 

            if (decision > 0) // if true, you also need to move vertically
            {
                // midpoints was outside the circle
                startY--; // y-- because the upper left corner is (0,0), so going up means, decrementing

                decision = decision + 4 * (startX - startY) + 10; 
            }
            else // midpoint was on the circle or inside it 
                decision = decision + 4 * startX + 6;

            plotHorizontalLines(startX, startY);
        }
    }

    void Renderer2D::drawTriangle(int x1, int y1, int x2, int y2, int x3, int y3, RGBA rgba)
    {
        drawLine(x1, y1, x2, y2, rgba);
        drawLine(x2, y2, x3, y3, rgba);
        drawLine(x1, y1, x3, y3, rgba);
    }

    void Renderer2D::fillBottomFlatTriangle(int x1, int y1, int x2, int y2, int x3, int y3, RGBA rgba)
    {
        float slope1 = (x2 - x1) / (y2 - y1);
        float slope2 = (x3 - x1) / (y3 - y1); 

        float currentX1 = x1;
        float currentX2 = x1;

        for (int startY = y1; startY <= y2; startY++) 
        {
            drawHorizontalLine(startY, (int)currentX1, (int)currentX2, rgba);
            currentX1 += slope1;
            currentX2 += slope2;
        }
    }

    void Renderer2D::fillTopFlatTriangle(int x1, int y1, int x2, int y2, int x3, int y3, RGBA rgba)
    {
        float slope1 = (x2 - x1) / (y2 - y1);
        float slope2 = (x3 - x1) / (y3 - y1); 

        float currentX1 = x1;
        float currentX2 = x1;

        for (int startY = y1; startY >= y2; startY--) 
        {
            drawHorizontalLine(startY, (int)currentX1, (int)currentX2, rgba);
            currentX1 -= slope1;
            currentX2 -= slope2;
        }
    }

    // https://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
    // https://gamedev.stackexchange.com/questions/178181/how-can-i-draw-filled-triangles-in-c
    // https://www.gabrielgambetta.com/computer-graphics-from-scratch/07-filled-triangles.html
    void Renderer2D::fillTriangle(int x1, int y1, int x2, int y2, int x3, int y3, RGBA rgba) // scanline triangle rasterization algorithm
    {
        std::vector<std::pair<int, int>> vertices;
        vertices.push_back(std::make_pair(x1, y1));
        vertices.push_back(std::make_pair(x2, y2));
        vertices.push_back(std::make_pair(x3, y3));

        std::sort(vertices.begin(), vertices.end(), [](const std::pair<int, int> a, const std::pair<int, int> b) { return a.second < b.second; });

        std::pair<int, int> v0 = vertices[0];
        std::pair<int, int> v1 = vertices[1];
        std::pair<int, int> v2 = vertices[2];

       // so you basically divide the triangle further into two triangles 
       // one with a flat bottom and one with a flat top
       // if it does not already have a flat top or bottom

        if (v1.second == v2.second)
        {
            fillBottomFlatTriangle(v0.first, v0.second, v1.first, v1.second, v2.first, v2.second, rgba);
        }
        else if (v0.second == v1.second)
        {
            fillTopFlatTriangle(v2.first, v2.second, v1.first, v1.second, v0.first, v0.second, rgba);
        }
        else 
        {
            // create a new vertex 
            std::pair<int, int> v3 = std::make_pair((int)(v0.first + ((float)(v1.second - v0.second) / (float)(v2.second - v1.second)) * (v2.first - v0.first)), v1.second);
            fillBottomFlatTriangle(v0.first, v0.second, v1.first, v1.second, v3.first, v3.second, rgba);
            fillTopFlatTriangle(v2.first, v2.second, v3.first, v3.second, v1.first, v1.second, rgba);
        }
    }
