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

        // Cache of vertices for this load operation
        std::vector<vec3d> verts;
        // Clear existing triangle data in this mesh object
        tris.clear();

        std::string line;
        int line_number = 0;
        bool load_error = false;

        while (std::getline(f, line)) {
            line_number++;
            std::stringstream ss(line);
            std::string command; // The command like 'v', 'f', 'vt', etc.
            ss >> command;

            if (command == "#" || command.empty()) {
                continue; // Ignore comments and empty lines
            } else if (command == "v") { // Geometric vertex
                vec3d v;
                if (!(ss >> v.x >> v.y >> v.z)) {
                     std::cerr << "WARNING: Line " << line_number << ": Malformed vertex data." << std::endl;
                     load_error = true; // Mark potential issue, but try to continue
                     continue;
                }
                verts.push_back(v);
            } else if (command == "f") { // Face element
                std::vector<int> face_indices;
                std::string vertex_data_str;

                while (ss >> vertex_data_str) {
                    size_t first_slash = vertex_data_str.find('/');
                    std::string v_index_str = vertex_data_str.substr(0, first_slash);

                    try {
                        int v_idx = std::stoi(v_index_str);

                        // Handle negative indices: relative to the *current* vertex count
                        if (v_idx < 0) {
                            // +1 because OBJ is 1-based, size() is count
                            v_idx = static_cast<int>(verts.size()) + v_idx + 1;
                        }

                        // Basic bounds check (1-based index must be valid)
                        if (v_idx <= 0 || static_cast<size_t>(v_idx) > verts.size()) {
                             std::cerr << "WARNING: Line " << line_number << ": Vertex index "
                                       << v_idx << " out of range (1 to " << verts.size() << ")." << std::endl;
                             face_indices.clear(); // Invalidate this face
                             load_error = true;
                             break; // Stop processing this face line
                        }

                        face_indices.push_back(v_idx); // Store the 1-based index for now

                    } catch (const std::invalid_argument& ia) {
                        std::cerr << "WARNING: Line " << line_number << ": Invalid face index format: '" << v_index_str << "'" << std::endl;
                        face_indices.clear(); load_error = true; break;
                    } catch (const std::out_of_range& oor) {
                        std::cerr << "WARNING: Line " << line_number << ": Face index out of range: '" << v_index_str << "'" << std::endl;
                        face_indices.clear(); load_error = true; break;
                    }
                } // End reading indices for one face line

                // --- Triangulate the face (simple fan triangulation) ---
                if (face_indices.size() >= 3) {
                    // Use the first vertex as the base for the fan
                    int idx0_1based = face_indices[0];
                    int idx0_0based = idx0_1based - 1; // Convert to 0-based for vector access

                    for (size_t i = 1; i < face_indices.size() - 1; ++i) {
                        int idx1_1based = face_indices[i];
                        int idx2_1based = face_indices[i + 1];
                        int idx1_0based = idx1_1based - 1;
                        int idx2_0based = idx2_1based - 1;

                        // Double check bounds just in case (though checked during read)
                        if (idx0_0based >= 0 && idx0_0based < verts.size() &&
                            idx1_0based >= 0 && idx1_0based < verts.size() &&
                            idx2_0based >= 0 && idx2_0based < verts.size())
                        {
                            tris.push_back({ verts[idx0_0based], verts[idx1_0based], verts[idx2_0based] });
                        } else {
                             std::cerr << "ERROR: Line " << line_number << ": Internal error - Invalid index during triangulation." << std::endl;
                             load_error = true;
                             // Optionally break out of triangulation loop for this face
                        }
                    }
                } else if (!face_indices.empty()) { // If we read indices but not enough for a triangle
                     std::cerr << "WARNING: Line " << line_number << ": Face defined with less than 3 vertices." << std::endl;
                     load_error = true;
                }
            }
            // Ignore other commands like vt, vn, o, g, s, usemtl, etc.

        } // End while getline

        f.close();

        if (load_error) {
             std::cerr << "NOTE: Errors or warnings occurred during OBJ load." << std::endl;
             // Decide if warnings should still result in 'true' or 'false'
             // return false; // Stricter: return false if any warning occurred
        }
        if (tris.empty() && !verts.empty()) {
             std::cerr << "WARNING: Loaded vertices but no valid faces found in " << sFilename << std::endl;
        } else if (tris.empty() && verts.empty() && !load_error) {
             std::cerr << "WARNING: File seems empty or contained no valid v/f data: " << sFilename << std::endl;
        }


        return !load_error; // Return true only if no errors were flagged (adjust based on desired strictness)
        // Or simply return true if the file was opened, regardless of content errors.
        // return true;
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
    void multiplyMatrixVector(vec3d &i, vec3d &o, mat4x4 &m);
    void update(float deltaTime);
};
