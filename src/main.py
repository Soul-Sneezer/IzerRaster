import IzerRaster


class CustomRenderer(IzerRaster.Renderer2D):
    def __init__(self, appName, width, height):
        super().__init__(appName, width, height)
        self.object_loaded = False

    def UserInit(self):
        # object path load
        obj_path = "suzanne.obj"
        self.object_loaded = self.loadObj(obj_path)
        print(self.object_loaded)
        self.theta = 0.0
        self.translate = 8.0
        self.renderMode = IzerRaster.RenderMode.SHADED_WIREFRAME

    def UserUpdate(self):
        events = self.poolInputEvents()

        for event in events:
            if event == None:
                continue
            elif event.type == "KEYDOWN" and event.key == IzerRaster.KEY_W:
                # adjust zoom out until the camera is fully implemented
                self.translate += 1.0
            elif event.type == "KEYDOWN" and event.key == IzerRaster.KEY_S:
                self.translate -= 1.0
            elif event.type == "KEYDOWN" and event.key == IzerRaster.KEY_ESCAPE:
                self.Quit()

        # Rotation overtime using glm
        self.theta += 4 * self.getDeltaTime()
        transform = IzerRaster.translate(
            IzerRaster.vec3(0.0, 0.0, self.translate))
        rotateX = IzerRaster.rotate(self.theta, IzerRaster.vec3(1.0, 0.0, 0.0))
        rotateY = IzerRaster.rotate(self.theta, IzerRaster.vec3(0.0, 1.0, 0.0))
        rotateZ = IzerRaster.rotate(self.theta, IzerRaster.vec3(0.0, 0.0, 1.0))

        new_mesh = self.applyRenderMatrix(
            transform * rotateY, self.object_loaded)

        self.drawObj(new_mesh)


renderer2D = CustomRenderer("Testing", 1920, 1080)
renderer2D.Init()
# renderer2D.setup_scene()
renderer2D.Run()
renderer2D.Quit()
