import os
import tkinter.ttk
from tkinter import messagebox
import customtkinter as tk
# Set appearance
tk.set_appearance_mode('dark')
tk.set_default_color_theme('blue')

# init window frame
window = tk.CTk()

# config window frame
window.minsize(width=1280,height=720)
window.title('Predict MaketPlace')

# cofig window grid
window.columnconfigure(0,weight=1)
window.columnconfigure(1,weight=2)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        path_1 = os.getcwd() + '\\fig1.jpeg'
        path_2 = os.getcwd() + '\\fig2.jpeg'

        if os.path.exists(path_1):
            os.remove(path_1)
        if os.path.exists(path_2):
            os.remove(path_2)
        window.destroy()

# Closing window
window.protocol("WM_DELETE_WINDOW", on_closing)

window.mainloop()
