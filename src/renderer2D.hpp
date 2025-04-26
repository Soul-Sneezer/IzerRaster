#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3_ttf/SDL_ttf.h>
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

struct triangle{
    vec3d p[3];
};

struct mesh {
    std::vector<triangle> tris;

    bool LoadFromObjectFile(const std::string& sFilename) {
        std::ifstream f(sFilename);
        if (!f.is_open()) {
            std::cerr << "ERROR: LoadFromObjectFile - Cannot open file: " << sFilename << std::endl;
            return false;
        }

        std::vector<vec3d> temp_verts;
        std::vector<vec2d> temp_texCoords;
        std::vector<vec3d> temp_normals;

        tris.clear();

        std::string line;
        int line_number = 0;
        bool load_error = false;

        while (std::getline(f, line)) {
            line_number++;
            std::stringstream ss(line);
            std::string command;
            ss >> command;

            if (command.empty() || command[0] == '#') {
                continue;

            } else if (command == "v") {
                vec3d v;
                if (!(ss >> v.x >> v.y >> v.z)) {
                    std::cerr << "WARNING: Line " << line_number << ": Malformed vertex data." << std::endl;
                    load_error = true; continue;
                }
                temp_verts.push_back(v);

            } else if (command == "vt") {//for texture to be done
                vec2d vt;
                if (!(ss >> vt.u >> vt.v)) {
                    std::cerr << "WARNING: Line " << line_number << ": Malformed texture coordinate data." << std::endl;
                    load_error = true; continue;
                }
                temp_texCoords.push_back(vt);

            } else if (command == "vn") {// vertex normal
                vec3d vn;
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
};

struct mat4x4{
    float m[4][4] = {0};
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
    mat4x4 matProj, matRotX, matRotZ, matTrans;
    float fTheta;
    vec3d vCamera;

    
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
    void multiplyMatrixVector(vec3d &i, vec3d &o, mat4x4 &m);
 
};
