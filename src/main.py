import IzerRaster


class CustomRenderer(IzerRaster.Renderer2D):
    def __init__(self, appName, width, height):
        super().__init__(appName, width, height)
        self.object_loaded = False
        # object path load
        obj_path = "suzanne.obj"
        self.object_loaded = self.loadObj(obj_path)
        print(self.object_loaded)

        # values for rotations
        self.theta = 0.0
        self.pitch_angle = 0.0
        self.free_rotate = False
        self.free_theta = 0

        # values for speed
        self.translate = 8.0
        self.move_step = 0.5
        self.rotation_step = 0.05
        self.renderMode = IzerRaster.RenderMode.SHADED_WIREFRAME

    def UserUpdate(self):
        events = self.poolInputEvents()

        for event in events:
            if event == None:
                continue
            elif event.type == "MOUSEWHEEL":  # wheel control
                if event.wheelY > 0:
                    print("Wheel Up")
                    self.translate -= 1.0
                elif event.wheelY < 0:
                    print("Wheel Down")
                    self.translate += 1.0
            elif event.type == "KEYDOWN":

                if event.key == IzerRaster.KEY_ESCAPE:
                    self.Quit()

                #  rotate up,down,left,right
                elif event.key == IzerRaster.KEY_W:
                    self.pitch_angle -= self.rotation_step
                elif event.key == IzerRaster.KEY_S:
                    self.pitch_angle += self.rotation_step
                elif event.key == IzerRaster.KEY_A:
                    self.theta -= self.rotation_step
                elif event.key == IzerRaster.KEY_D:
                    self.theta += self.rotation_step

                # free rotate showcase
                elif event.key == IzerRaster.KEY_R:
                    if self.free_rotate == False:
                        self.free_theta = self.theta
                        self.free_rotate = True
                    else:
                        self.theta = self.free_theta
                        self.pitch_angle = self.free_theta
                        self.free_rotate = False

        # the mathematic for rotatins and translation
        self.free_theta += self.getDeltaTime()
        transform = IzerRaster.translate(
            IzerRaster.vec3(0.0, 0.0, self.translate))

        if self.free_rotate == False:
            rotateX = IzerRaster.rotate(
                self.pitch_angle, IzerRaster.vec3(1.0, 0.0, 0.0))
            rotateZ = IzerRaster.rotate(
                self.theta, IzerRaster.vec3(0.0, 0.0, 1.0))
        else:
            rotateX = IzerRaster.rotate(
                self.free_theta, IzerRaster.vec3(1.0, 0.0, 0.0))
            rotateZ = IzerRaster.rotate(
                self.free_theta, IzerRaster.vec3(0.0, 0.0, 1.0))

        # Render our mathematic
        new_mesh = self.applyRenderMatrix(
            transform * rotateX * rotateZ, self.object_loaded)

        # Showing our mathematic :)
        self.drawObj(new_mesh)


renderer2D = CustomRenderer("Testing", 1920, 1080)
renderer2D.Init()
renderer2D.Run()
renderer2D.Quit()
