#include "renderer2D.hpp"

int main(int argc, char* argv[])
{
    const int SCREEN_WIDTH = 800;
    const int SCREEN_HEIGHT = 600;


    Renderer2D myRenderer("Testing", SCREEN_WIDTH, SCREEN_HEIGHT);
    myRenderer.Init();

    myRenderer.Run();

    myRenderer.Quit();
}
