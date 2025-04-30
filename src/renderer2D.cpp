#define SDL_MAIN_HANDLED
#include "renderer2D.hpp"
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>

    uint64_t Renderer2D::lastTime = 0;

    Renderer2D::Renderer2D(const std::string appName, uint16_t width, uint16_t height) : appName(appName), windowWidth(width), windowHeight(height)
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

        //for triangle projections and geometry
        float fNear = 0.1f;
        float fFar = 1000.0f;
        float fFov = 60.0f;
        float fAspectRatio = (float)windowWidth / (float)windowHeight;

        proj = glm::perspective(glm::radians(fFov), fAspectRatio, fNear, fFar);

        cameraPos = glm::vec4{0};
        
        isRunning = true;

        UserInit();
    }

    uint64_t Renderer2D::GetCurrentTime()
    {
        return SDL_GetTicks();
    }

    float Renderer2D::GetDeltaTime()
    {
        return (GetCurrentTime() - Renderer2D::lastTime) / 1000.0f;
    }

    //While Running
    void Renderer2D::Run()
    {
        Renderer2D::lastTime = GetCurrentTime();
        while (isRunning)
        {
            float deltaTime = GetDeltaTime();
            Renderer2D::lastTime = GetCurrentTime();
   
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

            Render();
        }
    }

    void Renderer2D::Render()
    {
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
        SDL_RenderClear(renderer);

        clearScreen();

        UserUpdate();
        
        SDL_UpdateTexture(screenTexture, nullptr, screenBuffer.data(),  windowWidth * sizeof(RGBA));

        SDL_RenderTexture(renderer, screenTexture, nullptr, nullptr);

        SDL_RenderPresent(renderer);
    }

    void Renderer2D::UserInit()
    {
    }

    void Renderer2D::UserUpdate()
    {
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

    void Renderer2D::drawPoint(uint16_t x, uint16_t y, RGBA rgba)
    {
        if (y < 0 || y >= windowHeight || x < 0 || x >= windowWidth)
            return;

        this->screenBuffer[y * windowWidth + x] = rgba; 
    }

    void Renderer2D::drawLine(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGBA rgba) // Brensenham's Line Algorithm
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

    void Renderer2D::drawRect(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGBA rgba)
    {
       drawLine(x1, y1, x2, y1, rgba);
       drawLine(x1, y1, x1, y2, rgba);
       drawLine(x2, y2, x1, y2, rgba);
       drawLine(x2, y2, x2, y1, rgba);
    }

    void Renderer2D::fillRect(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, RGBA rgba)
    {
        int startX = std::min(x1, x2);
        int startY = std::min(y1, y2);
        int endX = std::max(x1, x2);
        int endY = std::max(y1, y2);

        for(uint16_t y = startY; y < endY; y++)
        {
            drawHorizontalLine(y, startX, endX, rgba);
        }
    }


    // https://en.wikipedia.org/wiki/Midpoint_circle_algorithm 
    void Renderer2D::drawCircle(uint16_t x, uint16_t y, uint16_t radius, RGBA rgba)
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
    
    void Renderer2D::drawHorizontalLine(uint16_t y, uint16_t x1, uint16_t x2, RGBA rgba)
    {
        if (x1 > x2)
            std::swap(x1, x2);

        if (y < 0 || y >= windowHeight)
            return; 

        uint16_t startX = x1;
        uint16_t endX = std::min(windowWidth, x2 + 1);

        if (startX >= endX)
            return; 

        auto rowStart = screenBuffer.begin() + y * windowWidth + startX;
        std::fill(rowStart, rowStart + (endX - startX), rgba);
    }

    void Renderer2D::fillCircle(uint16_t x, uint16_t y, uint16_t radius, RGBA rgba)
    {
        if (radius < 0)
            return;

        if (radius == 0)
        {
            drawPoint(x, y, rgba);
            return;
        }
        
        // midpoint circle algorithm 
        uint16_t startX = 0;
        uint16_t startY = radius;

        int decision = 3 - 2 * radius;

        auto plotHorizontalLines = [&](uint16_t offsetX, uint16_t offsetY) // this is the only thing that changes from the drawCircle function
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

    void Renderer2D::drawTriangle(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t x3, uint16_t y3, RGBA rgba)
    {
        drawLine(x1, y1, x2, y2, rgba);
        drawLine(x2, y2, x3, y3, rgba);
        drawLine(x1, y1, x3, y3, rgba);
    }

    void Renderer2D::fillBottomFlatTriangle(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t x3, uint16_t y3, RGBA rgba)
    {
        float slope1 = (float)(x2 - x1) / (y2 - y1);
        float slope2 = (float)(x3 - x1) / (y3 - y1);

        float currentX1 = x1;
        float currentX2 = x1;

        for (uint16_t startY = y1; startY <= y2; startY++)
        {
            drawHorizontalLine(startY, (uint16_t)currentX1, (uint16_t)currentX2, rgba);
            currentX1 += slope1;
            currentX2 += slope2;
        }
    }

    void Renderer2D::fillTopFlatTriangle(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t x3, uint16_t y3, RGBA rgba)
    {
        float slope1 = (float)(x2 - x1) / (y2 - y1);
        float slope2 = (float)(x3 - x1) / (y3 - y1);

        float currentX1 = x1;
        float currentX2 = x1;

        for (uint16_t startY = y1; startY >= y2; startY--)
        {
            drawHorizontalLine(startY, (uint16_t)currentX1, (uint16_t)currentX2, rgba);
            currentX1 -= slope1;
            currentX2 -= slope2;
        }
    }

    // https://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
    // https://gamedev.stackexchange.com/questions/178181/how-can-i-draw-filled-triangles-in-c
    // https://www.gabrielgambetta.com/computer-graphics-from-scratch/07-filled-triangles.html
    void Renderer2D::fillTriangle(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t x3, uint16_t y3, RGBA rgba) // scanline triangle rasterization algorithm
    {
        std::vector<std::pair<uint16_t, uint16_t>> vertices;
        vertices.push_back(std::make_pair(x1, y1));
        vertices.push_back(std::make_pair(x2, y2));
        vertices.push_back(std::make_pair(x3, y3));

        std::sort(vertices.begin(), vertices.end(), [](const std::pair<uint16_t, uint16_t> a, const std::pair<uint16_t, uint16_t> b) {
                return (a.second < b.second) || (a.second == b.second && a.first < b.first); });

        std::pair<uint16_t, uint16_t> v0 = vertices[0];
        std::pair<uint16_t, uint16_t> v1 = vertices[1];
        std::pair<uint16_t, uint16_t> v2 = vertices[2];

       // so you basically divide the triangle further into two triangles
       // one with a flat bottom and one with a flat top
       // if it does not already have a flat top or bottom

        if ((v1.first - v0.first) * (v2.second - v0.second) == (v2.first - v0.first) * (v1.second - v0.second))
            return;

        if (v0.second == v1.second && v1.second == v2.second)
            return;

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
            std::pair<uint16_t, uint16_t> v3 = std::make_pair(v0.first + (uint16_t)((float)(v1.second - v0.second) / (v2.second - v0.second) * (v2.first - v0.first)), v1.second);
            fillBottomFlatTriangle(v0.first, v0.second, v1.first, v1.second, v3.first, v3.second, rgba);
            fillTopFlatTriangle(v2.first, v2.second, v3.first, v3.second, v1.first, v1.second, rgba);
        }
    }

    mesh Renderer2D::applyRenderMatrix(glm::mat4 mat, mesh objMesh)
    {
        mesh newMesh;
        for (auto tri : objMesh.tris)
        {
            triangle newTri;
            newTri.p[0] = mat * tri.p[0];
            newTri.p[1] = mat * tri.p[1];
            newTri.p[2] = mat * tri.p[2];

            newMesh.tris.push_back(newTri);
        }

        return newMesh;
    }

    mesh Renderer2D::loadObj(std::string path){
        mesh obj;
        obj.LoadFromObjectFile(path);
        
        return obj;
    }

    void Renderer2D::drawObj(mesh obj){
        simpleRender(obj);
    }

    void Renderer2D::simpleRender(mesh meshObj){
        std::vector<triangle> tria;

        triangle triProjected;

        //glm::mat4 transl = glm::translate(glm::vec3(0.0f,0.0f,8.0f));
       
        //mesh newMesh = applyRenderMatrix(transl * rotX * rotZ, meshObj);
           
        for(auto triTranslated: meshObj.tris){

            glm::vec3 normal, line1, line2;
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

            if(normal.x * (triTranslated.p[0].x - cameraPos.x) + normal.y * (triTranslated.p[0].y - cameraPos.y) + normal.z * (triTranslated.p[0].z - cameraPos.z) < 0.0f){
           
                triProjected.p[0] = proj * triTranslated.p[0];
                triProjected.p[1] = proj * triTranslated.p[1];
                triProjected.p[2] = proj * triTranslated.p[2];
                
                for(int i = 0; i < 3; i++) {
                    triProjected.p[i].x /= triProjected.p[i].w;
                    triProjected.p[i].y /= triProjected.p[i].w;
                    triProjected.p[i].z /= triProjected.p[i].w;
                }


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
            fillTriangle(
               static_cast<uint16_t>(triToRaster.p[0].x), static_cast<uint16_t>(triToRaster.p[0].y),
               static_cast<uint16_t>(triToRaster.p[1].x), static_cast<uint16_t>(triToRaster.p[1].y),
               static_cast<uint16_t>(triToRaster.p[2].x), static_cast<uint16_t>(triToRaster.p[2].y),
               RGBA(200,200,200,255)
           );
            drawTriangle(
               static_cast<uint16_t>(triToRaster.p[0].x), static_cast<uint16_t>(triToRaster.p[0].y),
               static_cast<uint16_t>(triToRaster.p[1].x), static_cast<uint16_t>(triToRaster.p[1].y),
               static_cast<uint16_t>(triToRaster.p[2].x), static_cast<uint16_t>(triToRaster.p[2].y),
               RGBA(100,100,100,200)
           );
        
       }
    }


    void Renderer2D::drawCube(){
        meshCube.tris = {
            {glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f), glm::vec4(1.0f, 1.0f, 0.0f, 1.0f)},
            {glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)},

            {glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)},
            {glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), glm::vec4(1.0f, 0.0f, 1.0f, 1.0f)},

            {glm::vec4(1.0f, 0.0f, 1.0f, 1.0f), glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), glm::vec4(0.0f, 1.0f, 1.0f, 1.0f)},
            {glm::vec4(1.0f, 0.0f, 1.0f, 1.0f), glm::vec4(0.0f, 1.0f, 1.0f, 1.0f), glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)},

            {glm::vec4(0.0f, 0.0f, 1.0f, 1.0f), glm::vec4(0.0f, 1.0f, 1.0f, 1.0f), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)},
            {glm::vec4(0.0f, 0.0f, 1.0f, 1.0f), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)},

            {glm::vec4(0.0f, 1.0f, 0.0f, 1.0f), glm::vec4(0.0f, 1.0f, 1.0f, 1.0f), glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)},
            {glm::vec4(0.0f, 1.0f, 0.0f, 1.0f), glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), glm::vec4(1.0f, 1.0f, 0.0f, 1.0f)},

            {glm::vec4(1.0f, 0.0f, 1.0f, 1.0f), glm::vec4(0.0f, 0.0f, 1.0f, 1.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)},
            {glm::vec4(1.0f, 0.0f, 1.0f, 1.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)},

        };
        simpleRender(meshCube);
    }

    bool mesh::LoadFromObjectFile(const std::string& sFilename) {
        std::ifstream f(sFilename);
        if (!f.is_open()) {
            std::cerr << "ERROR: LoadFromObjectFile - Cannot open file: " << sFilename << std::endl;
            return false;
        }

        std::vector<glm::vec4> temp_verts;
        std::vector<glm::vec2> temp_texCoords;
        std::vector<glm::vec3> temp_normals;

        tris.clear();

        std::string line;
        uint32_t line_number = 0;
        bool load_error = false;

        while (std::getline(f, line)) {
            line_number++;
            std::stringstream ss(line);
            std::string command;
            ss >> command;

            if (command.empty() || command[0] == '#') {
                continue;

            } else if (command == "v") {
                glm::vec4 v;
                if (!(ss >> v.x >> v.y >> v.z)) {
                    std::cerr << "WARNING: Line " << line_number << ": Malformed vertex data." << std::endl;
                    load_error = true; continue;
                }
                v.w = 1.0f;
                temp_verts.push_back(v);

            } else if (command == "vt") {//for texture to be done
                glm::vec2 vt;
                if (!(ss >> vt.x >> vt.y)) {
                    std::cerr << "WARNING: Line " << line_number << ": Malformed texture coordinate data." << std::endl;
                    load_error = true; continue;
                }
                temp_texCoords.push_back(vt);

            } else if (command == "vn") {// vertex normal
                glm::vec3 vn;
                if (!(ss >> vn.x >> vn.y >> vn.z)) {
                    std::cerr << "WARNING: Line " << line_number << ": Malformed normal data." << std::endl;
                    load_error = true; continue;
                }
                temp_normals.push_back(vn);

            } else if (command == "f") {
                std::vector<int> face_v_indices;
                std::string face_chunk_str;
                bool face_parse_error = false;

                while (ss >> face_chunk_str) { 
                    std::stringstream chunk_ss(face_chunk_str);
                    std::string segment;
                    int v_idx = 0, vt_idx = 0, vn_idx = 0; 
                    int part = 0; 
                    bool index_error = false;

                    while (std::getline(chunk_ss, segment, '/')) {
                        try {
                            if (!segment.empty()) {
                                int index = std::stoi(segment);

                                if (index < 0) {
                                    if (part == 0) index = static_cast<int>(temp_verts.size()) + index + 1;
                                    else if (part == 1) index = static_cast<int>(temp_texCoords.size()) + index + 1;
                                    else if (part == 2) index = static_cast<int>(temp_normals.size()) + index + 1;
                                }

                                if (part == 0) v_idx = index;
                                else if (part == 1) vt_idx = index;
                                else if (part == 2) vn_idx = index;

                            } 
                        } catch (const std::invalid_argument& ia) {
                             std::cerr << "WARNING: Line " << line_number << ": Invalid index number in face chunk '" << face_chunk_str << "'" << std::endl;
                             index_error = true; break;
                        } catch (const std::out_of_range& oor) {
                             std::cerr << "WARNING: Line " << line_number << ": Index number out of range in face chunk '" << face_chunk_str << "'" << std::endl;
                             index_error = true; break;
                        }
                        part++;
                        if (part > 2) break;
                    }

                    if (index_error) {
                        face_parse_error = true; break; 
                    }

                    if (v_idx <= 0 || static_cast<size_t>(v_idx) > temp_verts.size()) {
                         std::cerr << "WARNING: Line " << line_number << ": Vertex index " << v_idx
                                   << " out of range (1 to " << temp_verts.size() << ") in chunk '" << face_chunk_str << "'." << std::endl;
                         face_parse_error = true; break; // Stop processing this face line
                    }
                    face_v_indices.push_back(v_idx);
                } 

                if (face_parse_error) {
                    load_error = true;
                    continue; 
                }
                if (face_v_indices.size() >= 3) {
                    int idx0_1based = face_v_indices[0];

                    for (size_t i = 1; i < face_v_indices.size() - 1; ++i) {
                        int idx1_1based = face_v_indices[i];
                        int idx2_1based = face_v_indices[i + 1];

                        int idx0_0based = idx0_1based - 1;
                        int idx1_0based = idx1_1based - 1;
                        int idx2_0based = idx2_1based - 1;

                        if (idx0_0based >= 0 && idx0_0based < temp_verts.size() &&
                            idx1_0based >= 0 && idx1_0based < temp_verts.size() &&
                            idx2_0based >= 0 && idx2_0based < temp_verts.size())
                        {
                            tris.push_back({ temp_verts[idx0_0based], temp_verts[idx1_0based], temp_verts[idx2_0based] });
                            //here u can add features ;) 

                          
                        } else {
                             std::cerr << "ERROR: Line " << line_number << ": Internal error - Invalid index during triangulation." << std::endl;
                             load_error = true;
                             break; 
                        }
                    }
                } else if (!face_v_indices.empty()) {
                    std::cerr << "WARNING: Line " << line_number << ": Face defined with less than 3 valid vertices." << std::endl;
                    load_error = true;
                }
            }
        } 

        f.close();

        if (load_error) {
             std::cerr << "NOTE: Warnings or errors occurred during OBJ load of '" << sFilename << "'." << std::endl;
        }

        if (tris.empty() && !temp_verts.empty()) {
             std::cerr << "WARNING: Loaded vertices but no valid faces found in '" << sFilename << "'." << std::endl;
        } else if (tris.empty() && temp_verts.empty() && !load_error) {
             std::cerr << "WARNING: File seems empty or contained no valid v/f data: '" << sFilename << "'." << std::endl;
        }

        std::cout << "Finished loading '" << sFilename << "'. Triangles loaded: " << tris.size() << std::endl;
        return true;
    }

    std::optional<InputEvent> Renderer2D::detectInputEvent()
    {
         SDL_Event event;

        while (SDL_PollEvent(&event))
        {
            InputEvent input;

            switch(event.type)
            {
                case SDL_EVENT_KEY_DOWN:
                    input.type = "KEYDOWN";
                    input.key = event.key.key;
                    break; 
                case SDL_EVENT_KEY_UP:
                    input.type = "KEYUP";
                    input.key = event.key.key;
                    break;
                case SDL_EVENT_MOUSE_MOTION:
                    input.type = "MOUSEMOTION";
                    input.mouseX = event.motion.x;
                    input.mouseY = event.motion.y;
                    break;
                case SDL_EVENT_MOUSE_BUTTON_DOWN:
                    input.type = "MOUSEDOWN";
                    input.key = event.button.button;
                    input.mouseX = event.motion.x;
                    input.mouseY = event.motion.y;
                    break;
                case SDL_EVENT_MOUSE_BUTTON_UP:
                    input.type = "MOUSEUP";
                    input.key = event.button.button;
                    input.mouseX = event.motion.x;
                    input.mouseY = event.motion.y;
                    break;
                case SDL_EVENT_MOUSE_WHEEL:
                    input.type = "MOUSEWHEEL";
                    input.key = event.button.button;
                    input.mouseX = event.motion.x;
                    input.mouseY = event.motion.y;
                    break;
                default:
                    return std::nullopt;
            }

            return input;
        }

        return std::nullopt;;

    }

    std::vector<InputEvent> Renderer2D::poolInputEvents()
    {
        std::vector<InputEvent> events;
        SDL_Event event;

        while (SDL_PollEvent(&event))
        {
            InputEvent input;

            switch(event.type)
            {
                case SDL_EVENT_KEY_DOWN:
                    input.type = "KEYDOWN";
                    input.key = event.key.key;
                    break; 
                case SDL_EVENT_KEY_UP:
                    input.type = "KEYUP";
                    input.key = event.key.key;
                    break;
                case SDL_EVENT_MOUSE_MOTION:
                    input.type = "MOUSEMOTION";
                    input.mouseX = event.motion.x;
                    input.mouseY = event.motion.y;
                    break;
                case SDL_EVENT_MOUSE_BUTTON_DOWN:
                    input.type = "MOUSEDOWN";
                    input.key = event.button.button;
                    input.mouseX = event.motion.x;
                    input.mouseY = event.motion.y;
                    break;
                case SDL_EVENT_MOUSE_BUTTON_UP:
                    input.type = "MOUSEUP";
                    input.key = event.button.button;
                    input.mouseX = event.motion.x;
                    input.mouseY = event.motion.y;
                    break;
                case SDL_EVENT_MOUSE_WHEEL:
                    input.type = "MOUSEWHEEL";
                    input.key = event.button.button;
                    input.mouseX = event.motion.x;
                    input.mouseY = event.motion.y;
                    break;
                default:
                    continue;
            }

            events.push_back(input);
        }

        return events;
    }
