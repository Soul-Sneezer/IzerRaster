import IzerRaster

class CustomRenderer(IzerRaster.Renderer2D):
    def UserDraw(self):
        self.drawPoint(200, 200, IzerRaster.RGBA(250, 100, 0, 255))

renderer2D = CustomRenderer("Testing", 800, 600)

renderer2D.Init()
renderer2D.Run()
renderer2D.Quit()
