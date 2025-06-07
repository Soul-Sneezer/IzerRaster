import os
import tkinter as tk
from tkinter import filedialog, messagebox
import IzerRaster

class GuiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IzerRaster Loader")
        self.geometry("400x200")
        self.resizable(False, False)

        # Variabile care țin calea aleasă
        self.obj_path = None
        self.tex_path = None

        # --- Frame pentru alegerea OBJ ---
        frm_obj = tk.Frame(self, padx=10, pady=10)
        frm_obj.pack(fill="x")
        btn_obj = tk.Button(frm_obj, text="Select OBJ File", command=self.select_obj)
        btn_obj.pack(side="left")
        self.lbl_obj = tk.Label(frm_obj, text="(niciun fișier .obj ales)", anchor="w")
        self.lbl_obj.pack(side="left", padx=(10,0))

        # --- Frame pentru alegerea texturii ---
        frm_tex = tk.Frame(self, padx=10, pady=10)
        frm_tex.pack(fill="x")
        btn_tex = tk.Button(frm_tex, text="Select Texture (PNG/JPG)", command=self.select_tex)
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
            self.lbl_obj.config(text=os.path.basename(path))
        else:
            self.obj_path = None
            self.lbl_obj.config(text="(niciun fișier .obj ales)")
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
        if self.obj_path and self.tex_path:
            self.btn_start.config(state="normal")
        else:
            self.btn_start.config(state="disabled")

    def start_renderer(self):
        """Pornim renderer-ul IzerRaster cu căile selectate"""
        # Verificăm dacă fișierele există încă
        if not os.path.isfile(self.obj_path):
            messagebox.showerror("Eroare", f"Fișierul OBJ nu a fost găsit:\n{self.obj_path}")
            return
        if not os.path.isfile(self.tex_path):
            messagebox.showerror("Eroare", f"Fișierul textură nu a fost găsit:\n{self.tex_path}")
            return

        # Ascundem fereastra GUI și pornim SDL-ul
        self.withdraw()
        try:
            # Dimensiunea ferestrei SDL (poți ajusta după plac)
            width, height = 1280, 720
            renderer = CustomRenderer("IzerRaster Window", width, height, self.obj_path, self.tex_path)
        except Exception as e:
            messagebox.showerror("Eroare la inițializare", str(e))
            self.deiconify()
            return

        # Inițializare SDL/CUDA/Tekextură și rulare
        renderer.Init()
        renderer.Run()
        renderer.Quit()
        self.deiconify()
    #     self.destroy()  # închidem GUI după ce renderer-ul se oprește

class CustomRenderer(IzerRaster.Renderer2D):
    def __init__(self, appName, width, height, obj_path, tex_path):
        super().__init__(appName, width, height)

        # 1) Load .obj
        ok = self.loadObj(obj_path)
        if not ok:
            raise RuntimeError(f"Nu am putut încărca OBJ-ul:\n  {obj_path}")

        # 2) Load textură
        # tex = self.loadTexture(tex_path)
        # if tex is None:
        #     raise RuntimeError(f"Nu am putut încărca textura:\n  {tex_path}")
        # self.setTexture(tex)
            

        # 3) Setăm mod implicit de randare (poți schimba în WIREFRAME, SHADED etc.)
        self.mode = IzerRaster.RenderMode.WIREFRAME

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
