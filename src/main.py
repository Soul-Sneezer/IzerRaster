import IzerRaster

class CustomRenderer(IzerRaster.Renderer2D):
    def UserDraw(self):
        self.drawPoint(250, 200, IzerRaster.RGBA(250, 100, 0, 255))
        #self.drawLine(100, 100, 100, 400, IzerRaster.RGBA(0, 0, 255, 255))
        self.fillRect(540, 540, 660, 660, IzerRaster.RGBA(0, 0, 255, 255))
        #self.drawTriangle(100, 100, 150, 150, 100, 150, IzerRaster.RGBA(0, 255, 0, 100))
        self.drawCircle(600, 600, 60, IzerRaster.RGBA(0, 255, 0, 255))
        self.fillCircle(500, 500, 60, IzerRaster.RGBA(0, 255, 0, 255))
        self.fillRect(200, 200, 400, 400, IzerRaster.RGBA(0, 120, 180, 255))
        self.fillTriangle(400, 400, 300, 300, 400, 200, IzerRaster.RGBA(0, 255, 0, 255))

renderer2D = CustomRenderer("Testing", 1280, 720)
renderer2D.Init()
renderer2D.Run()
renderer2D.Quit()
