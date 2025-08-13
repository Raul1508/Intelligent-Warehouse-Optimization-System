import pandas as pd
import pickle


import tkinter as tk
from tkinter import ttk, PhotoImage
import cv2
import numpy as np
import pandas as pd

import customtkinter

from datetime import datetime, timedelta

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()

frame = customtkinter.CTkFrame(master=root)
frame.pack()

widgets_frame = customtkinter.CTkLabel(frame, text="Warehouse Design and Product Allocation")
widgets_frame.grid(row=0, column=0, padx=35, pady=22)

combo_list_1 = ["Autonomous Multilayer", "Normal Multilayer"]

def combobox_callback(choice):
    print("combobox dropdown clicked:", choice)

status_combobox_1 = customtkinter.CTkComboBox(master=widgets_frame,
                                     values=combo_list_1,
                                     command=combobox_callback)
status_combobox_1.grid(row=1, column=0, padx=5, pady=5,  sticky="ew")
status_combobox_1.set(combo_list_1[0])


from subprocess import call


def open_py_file(): 
    status = status_combobox_1.get()
    if(status == "Autonomous Multilayer"):
        call(["python","1stconcept_part9.py"])
    elif(status == "Normal Multilayer"):
        call(["python","1stconcept_part14.py"])

btn = customtkinter.CTkButton(widgets_frame, text="Start", command=open_py_file)
btn.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

def load(): 
    call(["python","Load.py"])

btn = customtkinter.CTkButton(widgets_frame, text="Load", command=load)
btn.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")


button_2 = customtkinter.CTkButton(widgets_frame, text="Exit", command=root.destroy)
button_2.grid(row=14, column=0, padx=5, pady=5, sticky="nsew")

root.mainloop()    