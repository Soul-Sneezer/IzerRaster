#define SDL_MAIN_HANDLED
#include "renderer2D.hpp"
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
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

        //for triangle projections and geometry
        float fNear = 0.1f;
        float fFar = 1000.0f;
        float fFov = 90.0f;
        float fAspectRatio = (float)windowHeight / (float)windowWidth;
        float fFovRad = 1.0/ tanf(fFov * 0.5f / 180.0f * 3.14159f);

        matProj.m[0][0] = fAspectRatio * fFovRad;
        matProj.m[1][1] = fFovRad;
        matProj.m[2][2] = fFar / (fFar - fNear);
        matProj.m[2][3] = 1.0f;
        matProj.m[3][2] = (-fFar * fNear) / (fFar - fNear);
        matProj.m[3][3] = 0.0f;

        vCamera = {0};
        
        isRunning = true;
    }

    //While Running
    void Renderer2D::Run()
    {
        Uint64 lastTime = SDL_GetTicks();
        while (isRunning)
        {
            Uint64 currentTime = SDL_GetTicks();
            float deltaTime = (currentTime - lastTime) / 1000.0f;
            lastTime = currentTime;
    
            HandleEvents();
            update(deltaTime); 
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

        clearScreen();

        UserDraw();
        
        SDL_UpdateTexture(screenTexture, nullptr, screenBuffer.data(),  windowWidth * sizeof(RGBA));

        SDL_RenderTexture(renderer, screenTexture, nullptr, nullptr);

        SDL_RenderPresent(renderer);
    }

    void Renderer2D::UserDraw()
    {
        //SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
        //drawLine(0, 0, windowWidth, windowHeight);
        
        // drawPoint(300, 400, {255, 255, 255, 255});
        // drawPoint(250, 320, {100, 120, 180, 255});
        // drawPoint(500, 200, {200, 50, 40, 200});
        drawCube();

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

    void Renderer2D::update(float deltaTime){
    fTheta += 1.0f * deltaTime; 

    matRotZ.m[0][0] = cosf(fTheta);
    matRotZ.m[0][1] = sinf(fTheta);
    matRotZ.m[1][0] = -sinf(fTheta);
    matRotZ.m[1][1] = cosf(fTheta);
    matRotZ.m[2][2] = 1;
    matRotZ.m[3][3] = 1;

    matRotX.m[0][0] = 1;
    matRotX.m[1][1] = cosf(fTheta * 0.5f);
    matRotX.m[1][2] = sinf(fTheta * 0.5f);
    matRotX.m[2][1] = -sinf(fTheta * 0.5f);
    matRotX.m[2][2] = cosf(fTheta * 0.5f);
    matRotX.m[3][3] = 1;

    }

    void Renderer2D::loadObj(std::string path){
        obj.LoadFromObjectFile("C:/Users/pasca/IzerRaster/obj/IronMan.obj");
        simpleRender(obj);
    }

    void Renderer2D::drawObj(){
        simpleRender(obj);
    }

    void Renderer2D::simpleRender(mesh meshObj){
        std::vector<triangle> tria;

        for(auto tri: meshObj.tris){
            triangle triProjected, triTranslated, triRotatedZ, triRotatedZX;

            multiplyMatrixVector(tri.p[0], triRotatedZ.p[0], matRotZ);
            multiplyMatrixVector(tri.p[1], triRotatedZ.p[1], matRotZ);
            multiplyMatrixVector(tri.p[2], triRotatedZ.p[2], matRotZ);

            multiplyMatrixVector(triRotatedZ.p[0], triRotatedZX.p[0], matRotX);
            multiplyMatrixVector(triRotatedZ.p[1], triRotatedZX.p[1], matRotX);
            multiplyMatrixVector(triRotatedZ.p[2], triRotatedZX.p[2], matRotX);

            //how far the object is
            triTranslated = triRotatedZX;
            triTranslated.p[0].z = triRotatedZX.p[0].z + 350.0f;
            triTranslated.p[1].z = triRotatedZX.p[1].z + 350.0f;
            triTranslated.p[2].z = triRotatedZX.p[2].z + 350.0f;

            vec3d normal, line1, line2;
            line1.x = triTranslated.p[1].x - triTranslated.p[0].x;
            line1.y = triTranslated.p[1].y - triTranslated.p[0].y;
            line1.z = triTranslated.p[1].z - triTranslated.p[0].z;

            line2.x = triTranslated.p[2].x - triTranslated.p[0].x;
            line2.y = triTranslated.p[2].y - triTranslated.p[0].y;
            line2.z = triTranslated.p[2].z - triTranslated.p[0].z;

            normal.x = line1.y * line2.z - line1.z * line2.y;
            normal.y = line1.z * line2.x - line1.x * line2.z;
            normal.z = line1.x * line2.y - line1.y * line2.x;

            float l = sqrtf(normal.x * normal.x + normal.y*normal.y + normal.z * normal.z);
            normal.x /= l;
            normal.y /= l;
            normal.z /= l;

            if(normal.x * (triTranslated.p[0].x - vCamera.x) + normal.y * (triTranslated.p[0].y - vCamera.y) + normal.z * (triTranslated.p[0].z - vCamera.z) < 0.0f){
                

                multiplyMatrixVector(triTranslated.p[0], triProjected.p[0], matProj);
                multiplyMatrixVector(triTranslated.p[1], triProjected.p[1], matProj);
                multiplyMatrixVector(triTranslated.p[2], triProjected.p[2], matProj);

                //Scale
                triProjected.p[0].x += 1.0f; triProjected.p[0].y += 1.0f;
				triProjected.p[1].x += 1.0f; triProjected.p[1].y += 1.0f;
				triProjected.p[2].x += 1.0f; triProjected.p[2].y += 1.0f;
				triProjected.p[0].x *= 0.5f * (float)windowWidth;
				triProjected.p[0].y *= 0.5f * (float)windowHeight;
				triProjected.p[1].x *= 0.5f * (float)windowWidth;
				triProjected.p[1].y *= 0.5f * (float)windowHeight;
				triProjected.p[2].x *= 0.5f * (float)windowWidth;
				triProjected.p[2].y *= 0.5f * (float)windowHeight;

                
                tria.push_back(triProjected);
            }

        }

        sort(tria.begin(), tria.end(), [](triangle &t1, triangle &t2)
		{
			float z1 = (t1.p[0].z + t1.p[1].z + t1.p[2].z) / 3.0f;
			float z2 = (t2.p[0].z + t2.p[1].z + t2.p[2].z) / 3.0f;
			return z1 > z2;
		});


        for (const auto& triToRaster : tria) {
            drawTriangle(
               static_cast<int>(triToRaster.p[0].x), static_cast<int>(triToRaster.p[0].y),
               static_cast<int>(triToRaster.p[1].x), static_cast<int>(triToRaster.p[1].y),
               static_cast<int>(triToRaster.p[2].x), static_cast<int>(triToRaster.p[2].y),
               RGBA(200,200,200,200)
           );
        
       }
    }


    void Renderer2D::drawCube(){
        meshCube.tris = {
            {0.0f, 0.0f, 0.0f,     0.0f, 1.0f,0.0f,       1.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f,     1.0f, 1.0f,0.0f,       1.0f, 0.0f, 0.0f},

            {1.0f, 0.0f, 0.0f,     1.0f, 1.0f,0.0f,       1.0f, 1.0f, 1.0f},
            {1.0f, 0.0f, 0.0f,     1.0f, 1.0f,1.0f,       1.0f, 0.0f, 1.0f},

            {1.0f, 0.0f, 1.0f,     1.0f, 1.0f,1.0f,       0.0f, 1.0f, 1.0f},
            {1.0f, 0.0f, 1.0f,     0.0f, 1.0f,1.0f,       0.0f, 0.0f, 1.0f},

            {0.0f, 0.0f, 1.0f,     0.0f, 1.0f,1.0f,       0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f,     0.0f, 1.0f,0.0f,       0.0f, 0.0f, 0.0f},

            {0.0f, 1.0f, 0.0f,     0.0f, 1.0f,1.0f,       1.0f, 1.0f, 1.0f},
            {0.0f, 1.0f, 0.0f,     1.0f, 1.0f,1.0f,       1.0f, 1.0f, 0.0f},

            {1.0f, 0.0f, 1.0f,     0.0f, 0.0f,1.0f,       0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 1.0f,     0.0f, 0.0f,0.0f,       1.0f, 0.0f, 0.0f},

        };
        simpleRender(meshCube);
    }

    void Renderer2D::multiplyMatrixVector(vec3d &i, vec3d &o, mat4x4 &m){
        o.x = i.x * m.m[0][0] + i.y * m.m[1][0] + i.z * m.m[2][0] + m.m[3][0];
        o.y = i.x * m.m[0][1] + i.y * m.m[1][1] + i.z * m.m[2][1] + m.m[3][1];
        o.z = i.x * m.m[0][2] + i.y * m.m[1][2] + i.z * m.m[2][2] + m.m[3][2];
        float w = i.x * m.m[0][3] + i.y * m.m[1][3] + i.z * m.m[2][3] + m.m[3][3];

        if(w != 0.0f){
            o.x /= w;
            o.y /= w;
            o.z /= w;
        }
    } 
