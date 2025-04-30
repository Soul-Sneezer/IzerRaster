# IzerRaster
IzerRaster is a rasterizer. What is a rasterizer? A rasterizer takes 3D models and projects them to a 2D surface, that 2D surface being the computer screen.

Currently the library has two main features: 
- it can load and render 3D models and basic 2D shapes(triangles, rectangles, circles)
- it can handle input events from the mouse or keyboard

The IzerRaster library has a core loop, named UserUpdate(), that runs once per frame. And a UserInit() function, which runs right after the basic Init() function that initializes everything. The user should override the "UserUpdate", "UserInit" function when defining the renderer class.

The drawing/rendering and input handling functions should be called from UserUpdate.

In order to render an object you need to:
- Load the object, using the _loadObj_ function. It takes a path as argument, and can only load .obj files. It then returns a _mesh_ object
- You can then modify the _mesh_ object using the _applyRenderMatrix_ function, which takes 2 arguments: a render matrix and a mesh
- Then you render the object using the _drawObj_ and passing the mesh as an argument
  
**IzerRaster** also has a few drawing functions:
- *drawPoint* - takes 3 arguments, the first is the X coordinate, the second is the Y coordinate and the last one is the color in RGBA format
- *drawLine* - takes 5 arguments, the first two are the X,Y coordinates of the first point of the line, then the X,Y coordinates of the second point of the line, and the color
- *drawRect* - takes 5 arguments, the first two are the X,Y coordinates of the lower left corner point, the following two are the X,Y coordinates of the upper right corner point, and then the color
- *drawTriangle* - takes 7 arguments, the first six are coordinates of the points that describe the triangle: X1,Y1, X2,Y2, X3,Y3, the last one is the color 
- *drawCircle* - takes 4 arguments, the first two are the X,Y coordinates of the center of the triangle, then the radius and lastly, the color

For almost every drawing function(with the exception of *drawPoint* and *drawLine*) there is an equivalent _fill_ function that takes the same arguments. As the name may say, a _fill_ function fills the shape, while the _draw_ functions draws the outline. The fill functions are the following: _fillRect_, _fillTriangle_, _fillCircle_.

Input handling is done in the following way:
- You call _poolInputEvents_ or _detectInputEvent_. The difference between them is that _poolInputEvents_ returns a list of all the events detected in a frame, while _detectInputEvent_ returns the last event detected in the frame
- Each input event has four fields: its type, the key that was pressed, and if it was a mouse event, the current coordinates of the mouse pointer. The user can then use this information to handle each event.

The event types are the following:
- *KEYUP* - detects if the user stopped pressing a key
- *KEYDOWN* - detects if the user pressed a key
- *MOUSEUP* - detects if the user stopped pressing a mouse key
- *MOUSEDOWN* - detects if the user pressed a mouse key
- *MOUSEWHEEL* - detects if the user moved the mouse wheel
- *MOUSEMOTION* - detects if the user moved the mouse

Most keyboard and mouse keys are supported and their values are marked with the following aliases:
- KEY_UNKNOWN
- KEY_RETURN
- KEY_ESCAPE
- KEY_BACKSPACE
- KEY_TAB
- KEY_SPACE
- KEY_EXCLAIM
- KEY_DBLAPOSTROPHE
- KEY_HASH
- KEY_DOLLAR
- KEY_PERCENT
- KEY_AMPERSAND
- KEY_APOSTROPHE
- KEY_LEFTPAREN
- KEY_RIGHTPAREN
- KEY_ASTERISK
- KEY_PLUS
- KEY_COMMA
- KEY_MINUS
- KEY_PERIOD
- KEY_SLASH
- KEY_LEFTBRACE
- KEY_PIPE
- KEY_RIGHTBRACE
- KEY_TILDE
- KEY_DELETE
- KEY_PLUSMINUS
- KEY_CAPSLOCK
- KEY_UP
- KEY_DOWN
- KEY_LEFT
- KEY_RIGHT
- KEY_1
- KEY_2
- KEY_3
- KEY_4
- KEY_5
- KEY_6
- KEY_7
- KEY_8
- KEY_9
- KEY_0
- KEY_A
- KEY_B
- KEY_C
- KEY_D
- KEY_E
- KEY_F
- KEY_G
- KEY_H
- KEY_I
- KEY_J
- KEY_K
- KEY_L
- KEY_M
- KEY_N
- KEY_O
- KEY_P
- KEY_Q
- KEY_R
- KEY_S
- KEY_T
- KEY_U
- KEY_V
- KEY_W
- KEY_X
- KEY_Y
- KEY_Z

### Setup
This project requires the following libraries:

- GLM: A header-only C++ mathematics library for graphics software.
- SDL3: The Simple DirectMedia Layer library for handling graphics, input, and audio.
- SDL3_ttf: An SDL extension for handling TrueType fonts.

Make sure to install these dependencies before building the project.

# Trello

We used Trello in order to manage and organise our tasks during the development of the application. This is the [link](https://trello.com/b/ZR0p4Yfg/rasterizer) to our backlog creation.

# User stories

User stories are short, simple descriptions of a feature told from the perspective of the person who desires the new capability, usually a user or customer of the system.

User story 1: As a Blender enthusiast, I want to import 3D models in common file formats so I can load and convert my designs without hassle.

User story 2: As a designer, I want to set the output resolution myself so I can ensure my images meet quality standards for client work.

User story 3: As a lighting artist, I want to tweak lighting settings so my rasterized scenes look dynamic and eye-catching.

User story 4: As a 3D animator, I want to switch between rendering modes to experiment with different visual styles for my portfolio.

User story 5: As a 3D enthusiast, I want to have simple command for different drawing shapes and have them displayed in a buffer for convenience.

User story 6: As a shader developer, I want to apply custom GLSL/HLSL shaders to add unique textures and effects to my models.

User story 7: As a virtual photographer, I want to adjust the camera’s position, angle, and zoom to frame my 3D scenes perfectly for renders.

User story 8: As a detail-focused tester, I want anti-aliasing enabled to eliminate jagged edges and ensure my final images look polished.

User story 9: As a freelancer with tight deadlines, I want a back face culling option before committing to a full render.

User story 10: As a game developer, I want GPU acceleration support to speed up rendering for complex models and large projects.

User story 11: As a newbie to 3D software, I want an intuitive interface with helpful tooltips so I can learn the app without getting overwhelmed.
