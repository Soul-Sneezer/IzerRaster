import IzerRaster


class CustomRenderer(IzerRaster.Renderer2D):
    def __init__(self, appName, width, height):
        super().__init__(appName, width, height)
        self.object_loaded = False

    def UserInit(self):
        obj_path = "suzanne.obj"
        self.object_loaded = self.loadObj(obj_path)
        print(self.object_loaded)
        self.theta = 0.0
        self.translate = 8.0
    def UserUpdate(self):
        events = self.poolInputEvents()

        for event in events:
            if event == None:
                continue
            elif event.type == "KEYDOWN" and event.key == IzerRaster.KEY_W:
                self.translate += 1.0 
            elif event.type == "KEYDOWN" and event.key == IzerRaster.KEY_ESCAPE:
                self.Quit()
        
        self.theta += self.getDeltaTime()
        transform = IzerRaster.translate(IzerRaster.vec3(0.0, 0.0, self.translate))
        rotateX = IzerRaster.rotate(self.theta, IzerRaster.vec3(1.0, 0.0, 0.0))
        rotateZ = IzerRaster.rotate(self.theta, IzerRaster.vec3(0.0, 0.0, 1.0))

        new_mesh = self.applyRenderMatrix(transform * rotateX * rotateZ, self.object_loaded)

        #self.drawPoint(250, 200, IzerRaster.RGBA(250, 100, 0, 0))
        #self.drawLine(100, 100, 100, 400, IzerRaster.RGBA(0, 0, 255, 255))
        #self.fillRect(540, 540, 660, 660, IzerRaster.RGBA(0, 0, 255, 255))
        #self.drawTriangle(100, 100, 150, 150, 100, 150, IzerRaster.RGBA(0, 255, 0, 100))
        #self.drawCircle(600, 600, 60, IzerRaster.RGBA(0, 255, 0, 255))
        #self.fillCircle(500, 500, 60, IzerRaster.RGBA(0, 255, 0, 255))
        #self.fillRect(200, 200, 400, 400, IzerRaster.RGBA(0, 120, 180, 255))
        #self.fillTriangle(400, 400, 300, 300, 400, 200, IzerRaster.RGBA(200, 200, 200, 200))
 
        self.drawObj(new_mesh)

        

renderer2D = CustomRenderer("Testing", 1920, 1080)
renderer2D.Init()
#renderer2D.setup_scene()
renderer2D.Run()
renderer2D.Quit()
