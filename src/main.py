import os
import tkinter as tk
from tkinter import filedialog, messagebox
import IzerRaster

class GuiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IzerRaster Loader")
        self.geometry("400x300")  # Increased height for new button
        self.resizable(False, False)

        # Variabile care țin calea aleasă
        self.obj_path = None
        self.stl_path = None
        self.tex_path = None

        # --- Frame pentru alegerea OBJ ---
        frm_obj = tk.Frame(self, padx=10, pady=10)
        frm_obj.pack(fill="x")
        btn_obj = tk.Button(frm_obj, text="Select OBJ File", command=self.select_obj)
        btn_obj.pack(side="left")
        self.lbl_obj = tk.Label(frm_obj, text="(niciun fișier .obj ales)", anchor="w")
        self.lbl_obj.pack(side="left", padx=(10,0))

        # --- Frame pentru alegerea STL ---
        frm_stl = tk.Frame(self, padx=10, pady=10)
        frm_stl.pack(fill="x")
        btn_stl = tk.Button(frm_stl, text="Select STL File", command=self.select_stl)
        btn_stl.pack(side="left")
        self.lbl_stl = tk.Label(frm_stl, text="(niciun fișier .stl ales)", anchor="w")
        self.lbl_stl.pack(side="left", padx=(10,0))

        # --- Frame pentru alegerea texturii ---
        frm_tex = tk.Frame(self, padx=10, pady=10)
        frm_tex.pack(fill="x")
        btn_tex = tk.Button(frm_tex, text="Select Texture (PNG/JPG) - Optional for STL", command=self.select_tex)
        btn_tex.pack(side="left")
        self.lbl_tex = tk.Label(frm_tex, text="(niciun fișier .png/.jpg ales)", anchor="w")
        self.lbl_tex.pack(side="left", padx=(10,0))

        # --- Frame pentru butonul Start ---
        frm_start = tk.Frame(self, padx=10, pady=20)
        frm_start.pack(fill="x")
        self.btn_start = tk.Button(frm_start, text="Start Renderer", state="disabled", command=self.start_renderer)
        self.btn_start.pack()

    def select_obj(self):
        """Deschide dialog pentru a alege fișierul .obj"""
        path = filedialog.askopenfilename(
            title="Alege fișierul .obj",
            filetypes=[("OBJ files", "*.obj")])
        if path:
            self.obj_path = path
            self.stl_path = None  # Clear STL selection when OBJ is selected
            self.lbl_obj.config(text=os.path.basename(path))
            self.lbl_stl.config(text="(niciun fișier .stl ales)")
        else:
            self.obj_path = None
            self.lbl_obj.config(text="(niciun fișier .obj ales)")
        self._update_start_button()

    def select_stl(self):
        """Deschide dialog pentru a alege fișierul .stl"""
        path = filedialog.askopenfilename(
            title="Alege fișierul .stl",
            filetypes=[("STL files", "*.stl")])
        if path:
            self.stl_path = path
            self.obj_path = None  # Clear OBJ selection when STL is selected
            self.lbl_stl.config(text=os.path.basename(path))
            self.lbl_obj.config(text="(niciun fișier .obj ales)")
        else:
            self.stl_path = None
            self.lbl_stl.config(text="(niciun fișier .stl ales)")
        self._update_start_button()

    def select_tex(self):
        path = filedialog.askopenfilename(
            title="Alege fișierul PNG/JPG",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("PNG files", "*.png"),
                ("JPG/JPEG files", "*.jpg *.jpeg")
                        ]
            )
        if path:
            self.tex_path = path
            self.lbl_tex.config(text=os.path.basename(path))
        else:
            self.tex_path = None
            self.lbl_tex.config(text="(niciun fișier .png/.jpg ales)")
        self._update_start_button()

    def _update_start_button(self):
        """Activează/dezactivează butonul Start conform selecțiilor făcute"""
        # For OBJ files: require both model and texture
        # For STL files: only require model (texture is optional)
        if self.obj_path and self.tex_path:
            self.btn_start.config(state="normal")
        elif self.stl_path:  # STL only needs the model file, texture is optional
            self.btn_start.config(state="normal")
        else:
            self.btn_start.config(state="disabled")

    def start_renderer(self):
        """Pornim renderer-ul IzerRaster cu căile selectate"""
        # Determine which model file we're using
        model_path = self.obj_path if self.obj_path else self.stl_path
        model_type = "OBJ" if self.obj_path else "STL"
        
        # Verificăm dacă fișierele există încă
        if not os.path.isfile(model_path):
            messagebox.showerror("Eroare", f"Fișierul {model_type} nu a fost găsit:\n{model_path}")
            return
        
        # For OBJ files, texture is required
        if model_type == "OBJ" and not self.tex_path:
            messagebox.showerror("Eroare", "Fișierele OBJ necesită o textură!")
            return
            
        # Check texture exists if provided
        if self.tex_path and not os.path.isfile(self.tex_path):
            messagebox.showerror("Eroare", f"Fișierul textură nu a fost găsit:\n{self.tex_path}")
            return

        # Ascundem fereastra GUI și pornim SDL-ul
        self.withdraw()
        try:
            # Dimensiunea ferestrei SDL (poți ajusta după plac)
            width, height = 1280, 720
            renderer = IzerRaster.Renderer2D.Instance("IzerRaster Window", width , height)
            renderer.setCUDA(False)
            renderer.Init()

            # Load the appropriate model type
            if model_type == "OBJ":
                renderer.loadObj(model_path)
            else:  # STL
                renderer.loadStl(model_path)
            
            # Load texture if provided
            if self.tex_path:
                tex = renderer.loadTexture(self.tex_path)
                if tex is None:
                    print(f"Warning: Could not load texture: {self.tex_path}")
                    renderer.renderMode = IzerRaster.RenderMode.TEXTURED_WIREFRAME
                else:
                    renderer.setTexture(tex)
                    renderer.renderMode = IzerRaster.RenderMode.TEXTURED_WIREFRAME
            else:
                # No texture - use shaded wireframe mode
                renderer.renderMode = IzerRaster.RenderMode.SHADED_WIREFRAME
            
        except Exception as e:
            messagebox.showerror("Eroare la inițializare", str(e))
            self.deiconify()
            return

        # Inițializare SDL/CUDA/Tekextură și rulare
        renderer.Run()
        renderer.Quit()
        self.deiconify()

class CustomRenderer(IzerRaster.Renderer2D):
    def __init__(self, appName, width, height, model_path, tex_path, model_type):
        super().__init__(appName, width, height)

        # Store model information
        self.model_path = model_path
        self.model_type = model_type
        self.tex_path = tex_path

        # Inițializare parametri de transformare
        self.theta = 0.0
        self.pitch_angle = 0.0
        self.free_rotate = False
        self.free_theta = 0.0
        self.translate = 8.0
        self.move_step = 0.5
        self.rotation_step = 0.05

if __name__ == "__main__":
    app = GuiApp()
    app.mainloop()