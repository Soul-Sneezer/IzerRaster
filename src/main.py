import IzerRaster

class CustomRenderer(IzerRaster.Renderer2D):
    def __init__(self, appName, width, height, gpu=False):
        super().__init__(appName, width, height, gpu)
        self.gpu = gpu
        self.object_loaded = False
        obj_path = "/home/alex/IzerRaster/obj/suzanne.obj"
        self.object_loaded = self.loadObj(obj_path)
 

        # Initialize transformation and control parameters
        self.theta = 0.0
        self.pitch_angle = 0.0
        self.free_rotate = False
        self.free_theta = 0.0

        self.translate = 8.0
        self.move_step = 0.5
        self.rotation_step = 0.05
        self.renderMode = IzerRaster.RenderMode.SHADED_WIREFRAME

# Example usage:
# Create renderer in GPU mode (use CUDA pipeline)
renderer = CustomRenderer("Testing", 1920, 1080, gpu=True)
renderer.Init()
renderer.Run()
renderer.Quit()
