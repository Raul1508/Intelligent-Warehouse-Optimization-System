import tkinter as tk
from tkinter import ttk, PhotoImage
import cv2
import numpy as np
import pandas as pd
import psutil

import customtkinter

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")



import os

def is_python_file_open(file_name):
    """
    Check if a Python file is currently open as a running process.
    Compatible with both Windows and Linux.
    """
    file_name = os.path.basename(file_name)  # Get only the filename
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            process_name = proc.info['name'].lower()
            cmdline = proc.info['cmdline']

            # Check if the process is Python
            if "python" in process_name or "py" in process_name:
                if cmdline and any(file_name in arg for arg in cmdline):
                    return True  # Python file is running

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass  # Skip inaccessible processes

    return False

file_to_check = "Load.py"

if is_python_file_open(file_to_check):
    print("Loaded")



# def is_python_file_open(file_path):
#     for proc in psutil.process_iter():
#         try:
#             if proc.name().lower() == "python" and file_path in proc.cmdline():
#                 return True
#         except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
#             pass
#     return False

# file_path_to_check = "Load.py"

# if is_python_file_open(file_path_to_check):

#     print("Loaded")

else:

    s = {'layout_num':[],'layout_width':[],'layout_long':[]}
    dataframe_2 = pd.DataFrame(s)
    dataframe_2.to_csv("GUI_part4_layout.csv",index = False)

    d = {'layout_num':[],'ID':[],'layout_width':[],'layout_long':[],'width_p':[],'long_p':[],'looseness_corridor_w':[],'looseness_corridor_l':[],'collide_I_w':[],'collide_I_l':[],'collide_II_w':[],'collide_II_l':[],'result_distance_I':[],'result_distance_II':[],'min_collide_result':[],'min_distance_w':[],'min_distance_l':[],'remains_def_w':[],'remains_def_l':[],'remains_result_distance':[],'pos_width_p':[],'pos_long_p':[],'final_pos_width_p':[],'final_pos_long_p':[],'ID_Total':[]}
    dataframe = pd.DataFrame(d)
    dataframe.to_csv("GUI_part4_distance.csv",index = False)

    t = {'layout_num':[],'remains_w':[],'remains_l':[],'result_distance':[]}
    dataframe_1 = pd.DataFrame(t)
    dataframe_1.to_csv("GUI_part4_remains.csv",index = False)

    v = {'Artifice':[1]}
    dataframe = pd.DataFrame(v)
    dataframe.to_csv("Artifice_3.csv",index = False)

    u = {'thislist':[]}
    dataframe_u = pd.DataFrame(u)
    dataframe_u.to_csv("Images_save_1.csv",index = False)


root = customtkinter.CTk()

frame = customtkinter.CTkFrame(master=root)
frame.pack()

widgets_frame = customtkinter.CTkLabel(frame, text="GUI Automatic Warehouse Design")
widgets_frame.grid(row=0, column=0, padx=35, pady=22)


# my_username = customtkinter.CTkEntry(widgets_frame)
# my_username.insert(0, "Username")
# my_username.bind("<FocusIn>", lambda e: my_username.delete('0', 'end'))
# my_username.grid(row=1, column=0, padx=5, pady=(0, 5), sticky="ew")

Width = customtkinter.CTkEntry(widgets_frame)
Width.insert(0, "Width Warehouse")
# Width.bind("<FocusIn>", lambda e: Width.delete('0', 'end'))
Width.grid(row=2, column=0, padx=5, pady=(0, 5), sticky="ew")

Long = customtkinter.CTkEntry(widgets_frame)
Long.insert(0, "Length Warehouse")
# Long.bind("<FocusIn>", lambda e: Long.delete('0', 'end'))
Long.grid(row=3, column=0, padx=5, pady=(0, 5), sticky="ew")  

def insert_layout():

    width = int(Width.get())
    long = int(Long.get())
    blank_image = np.zeros((long, width, 3), dtype=np.uint8)
    blank_image[:] = (255, 255, 255)  # White color in BGR format


    dataframe_x = pd.read_csv("Artifice_3.csv")
    i = dataframe_x['Artifice'].values[0]
    
    dataframe = pd.read_csv("GUI_part4_layout.csv") 
    dataframe = pd.DataFrame({'layout_num': [i],'layout_width': [width],'layout_long': [long]})
    dataframe.to_csv('GUI_part4_layout.csv', mode='a', index=False, header=False)
    
    dataframe_x = dataframe_x.drop(['Artifice'], axis=1)
    dataframe_x.insert(0,"Artifice",[i+1],False)
    dataframe_x.to_csv("Artifice_3.csv",index = False,header=True)

    dataframe_2 = pd.read_csv("GUI_part4_layout.csv")
    layout_num = dataframe_2['layout_num'].tolist()
    combo["values"] = layout_num


    door = cv2.imread('door1.png')
    img1 = cv2.resize(door,(80,80))
    # x_end = width-int(width/10) + img1.shape[0]
    x_end = long-80 + img1.shape[0]
    y_end = img1.shape[1]
    # y_end = long-int(long/10) + img1.shape[1]
    # blank_image[width-int(width/10):x_end,long-int(long/10):y_end]=img1 
    blank_image[long-80:x_end,0:y_end]=img1 

    # Layout limit
    start_point_line = (0, 0)
    end_point_line = (width, long)
    blue_color = (255, 0, 0)  # Blue color in BGR format
    thickness_line = 5
    img_s = cv2.rectangle(blank_image, start_point_line, end_point_line, blue_color, thickness_line)
    image_saved = cv2.imwrite(f'Layer_{i}.png', img_s)

    afx = pd.DataFrame({'thislist':[f'Layer_{i}.png']})
    dfx = pd.read_csv("Images_save_1.csv")  
    df2x = pd.concat([afx,dfx], ignore_index=True) 
    df2x.to_csv('Images_save_1.csv',index = False,header=True)

    cv2.imshow("Image with Drawing", blank_image)
    cv2.setWindowTitle("Image with Drawing",f'Layer {i}')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

button = customtkinter.CTkButton(widgets_frame, text="Insert Layout", command=insert_layout)
button.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")

# Define the style for combobox widget
style= ttk.Style()
style.theme_use('clam')
style.configure("TCombobox", fieldbackground= "orange", background= "white")


widgets_frame = customtkinter.CTkLabel(frame, text="Layer N°")
widgets_frame.grid(row=5, column=0, padx=5, pady=5)


combo = ttk.Combobox(widgets_frame)
combo.grid(row=6, column=0, padx=5, pady=5,  sticky="ew")


dataframe_2 = pd.read_csv("GUI_part4_layout.csv")
layout_num = dataframe_2['layout_num'].tolist()

combo["values"] = layout_num


def visualize():

    i = int(combo.get())
    # Load the image
    image = cv2.imread(f'Layer_{i}.png')
    # Display the image in a window
    cv2.imshow('Image', image)
    # Wait for a key press
    cv2.setWindowTitle("Image",f'Layer {i}')
    cv2.waitKey(0)
    # Close window
    cv2.destroyAllWindows()

button_x = customtkinter.CTkButton(widgets_frame, text="Visualize Layout", command=visualize)
button_x.grid(row=7, column=0, padx=5, pady=5, sticky="nsew")


Width_p = customtkinter.CTkEntry(widgets_frame)
Width_p.insert(0, "Width of Area")
# Width_p.bind("<FocusIn>", lambda e: Width_p.delete('0', 'end'))
Width_p.grid(row=8, column=0, padx=5, pady=(0, 5), sticky="ew")

Long_p = customtkinter.CTkEntry(widgets_frame)
Long_p.insert(0, "Length of Area")
# Long_p.bind("<FocusIn>", lambda e: Long_p.delete('0', 'end'))
Long_p.grid(row=9, column=0, padx=5, pady=(0, 5), sticky="ew")

Corridor = customtkinter.CTkEntry(widgets_frame)
Corridor.insert(0, "Corridor Width")
# Corridor.bind("<FocusIn>", lambda e: Long_p.delete('0', 'end'))
Corridor.grid(row=10, column=0, padx=5, pady=(0, 5), sticky="ew")


def area():

    layout_num = int(combo.get())
    width_p = int(Width_p.get())
    long_p = int(Long_p.get())
    looseness_corridor_w = int(Corridor.get())
    looseness_corridor_l = int(Corridor.get())
        
    dataframe_2 = pd.read_csv("GUI_part4_layout.csv") 
    layout_width = dataframe_2['layout_width'].values[0]    
    layout_long = dataframe_2['layout_long'].values[0]

    dataframe_x = pd.read_csv("GUI_part4_distance.csv") 
    try:
        ID = dataframe_x[dataframe_x['layout_num']==layout_num]['ID'].max()  
        ID_Total = 1 
        if(ID>0):
            ID = ID
            ID_Prev_verif = dataframe_x['ID_Total'].values[0] 
            ID_Prev = dataframe_x[dataframe_x['layout_num']==(layout_num-1)]['ID_Total'].max() 
            if(ID_Prev>0)&(ID_Prev == ID_Prev_verif):
                ID_Total = ID_Prev + 1 
            elif (ID_Prev>0)&(ID_Prev != ID_Prev_verif):
                ID_Total = dataframe_x['ID_Total'].max()
                ID_Total = ID_Total + 1
            else:
                ID_Total = ID + 1
        else:
            ID = 0 
            ID_Prev_verif = dataframe_x['ID_Total'].values[0] 
            ID_Prev = dataframe_x[dataframe_x['layout_num']==(layout_num-1)]['ID_Total'].max() 
            if(ID_Prev>0)&(ID_Prev == ID_Prev_verif):
                ID_Total = ID_Prev + 1
            else:
                ID_Total = 1
    except IndexError:
        pass
    
    
    if ID == 0:

        ID = 1
    
        collide_I_w = 80 + 5
        collide_I_l = int(layout_long - long_p)
        collide_II_w = 0
        collide_II_l = int(layout_long -80 -5 - long_p) 
        result_distance_I = int(((collide_I_w)**2+(layout_long-collide_I_l)**2)**(1/2))
        result_distance_II = int(((collide_II_w)**2+(layout_long-collide_II_l)**2)**(1/2))

        if result_distance_I < result_distance_II:
            min_collide_result = "Type_I"
        else:
            min_collide_result = "Type_II"
        
        if result_distance_I < result_distance_II:
            min_distance_w = collide_I_w
        else:
            min_distance_w = collide_II_w

        if result_distance_I < result_distance_II:
            min_distance_l = collide_I_l
        else:
            min_distance_l = collide_II_l

        if result_distance_I < result_distance_II:
            remains_w = collide_II_w
        else:
            remains_w = collide_I_w
        
        if result_distance_I < result_distance_II:
            remains_l = collide_II_l
        else:
            remains_l = collide_I_l

        result_distance = int(((remains_w)**2+(layout_long-remains_l)**2)**(1/2))

        remains_def_w = remains_w
        remains_def_l = remains_l
        remains_result_distance = result_distance    
        
        pos_width_p = min_distance_w
        pos_long_p = min_distance_l
        final_pos_width_p = pos_width_p + width_p
        final_pos_long_p = pos_long_p + long_p

        if (((collide_I_w + width_p) > layout_width) or (collide_II_l<0)):
            blank_image = np.zeros((layout_long, layout_width, 3), dtype=np.uint8)
            blank_image[:] = (64, 64, 64)  # White color in BGR format
            # Layout limit
            start_point_line = (0, 0)
            end_point_line = (layout_width, layout_long)
            blue_color = (255, 0, 0)  # Blue color in BGR format
            thickness_line = 5
            img_s = cv2.rectangle(blank_image, start_point_line, end_point_line, blue_color, thickness_line)
            image_saved = cv2.imwrite('greetings.png', img_s)
            text = f"LIMIT REACHED"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (0, 0, 0)  # Black color in BGR format
            font_thickness = 2
            text_position = (int((layout_width/2-75)), int((layout_long/2+5)))
            layout = cv2.imread('greetings.png')
            img_s= cv2.putText(
                layout, text, text_position, font, font_scale, font_color, font_thickness
            )
            image_saved = cv2.imwrite(f'greetings.png', img_s)
            afx = pd.DataFrame({'thislist':[f'greetings.png']})
            dfx = pd.read_csv("Images_save_1.csv")  
            df2x = pd.concat([afx,dfx], ignore_index=True) 
            df2x.to_csv('Images_save_1.csv',index = False,header=True)
            cv2.imshow("Image with Drawing", layout)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return print("LIMIT REACHED")
        else:
            pass

        af = pd.DataFrame({'layout_num':[layout_num],'ID':[ID],'layout_width':[layout_width],'layout_long':[layout_long],'width_p':[width_p],'long_p':[long_p],'looseness_corridor_w':[looseness_corridor_w],'looseness_corridor_l':[looseness_corridor_l],'collide_I_w':[collide_I_w],'collide_I_l':[collide_I_l],'collide_II_w':[collide_II_w],'collide_II_l':[collide_II_l],'result_distance_I':[result_distance_I],'result_distance_II':[result_distance_II],'min_collide_result':[min_collide_result],'min_distance_w':[min_distance_w],'min_distance_l':[min_distance_l],'remains_def_w':[remains_def_w],'remains_def_l':[remains_def_l],'remains_result_distance':[remains_result_distance],'pos_width_p':[pos_width_p],'pos_long_p':[pos_long_p],'final_pos_width_p':[final_pos_width_p],'final_pos_long_p':[final_pos_long_p],'ID_Total':[ID_Total]})
        df = pd.read_csv("GUI_part4_distance.csv")  
        df2 = pd.concat([af,df], ignore_index=True) 
        df2.to_csv('GUI_part4_distance.csv',index = False,header=True)

        start_point_rect = (pos_width_p, pos_long_p)
        end_point_rect = (final_pos_width_p, final_pos_long_p)
        red_color = (0, 0, 255)  # Red color in BGR format
        thickness_rect = 2
        layout = cv2.imread(f'Layer_{layout_num}.png')
        img_s= cv2.rectangle(layout, start_point_rect, end_point_rect, red_color, thickness_rect)
        image_saved = cv2.imwrite(f'Layer_{layout_num}.png', img_s)
        # Add text to the Area
        text = f"Area {ID_Total}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 0, 0)  # Black color in BGR format
        font_thickness = 2
        text_position = (int((pos_width_p+width_p/2-30)), int((pos_long_p+long_p/2+5)))
        layout = cv2.imread(f'Layer_{layout_num}.png')
        img_s= cv2.putText(
            layout, text, text_position, font, font_scale, font_color, font_thickness
        )
        image_saved = cv2.imwrite(f'Layer_{layout_num}.png', img_s)
        cv2.imshow("Image with Drawing", layout)
        cv2.setWindowTitle("Image with Drawing",f'Layer {layout_num}')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif ID == 1:

        ID = 2
    
        dataframe = pd.read_csv("GUI_part4_distance.csv") 
        pos_width_p = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==1]['pos_width_p'].max()
        pos_long_p = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==1]['pos_long_p'].max()
        final_pos_width_p = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==1]['final_pos_width_p'].max()
        final_pos_long_p = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==1]['final_pos_long_p'].max()

        collide_I_w = final_pos_width_p + looseness_corridor_w
        collide_I_l = pos_long_p  
        collide_II_w = pos_width_p
        collide_II_l = pos_long_p - looseness_corridor_l - long_p
        
        if (((collide_I_w + width_p) > layout_width) and (collide_II_l<0)):
            blank_image = np.zeros((layout_long, layout_width, 3), dtype=np.uint8)
            blank_image[:] = (64, 64, 64)  # White color in BGR format
            # Layout limit
            start_point_line = (0, 0)
            end_point_line = (layout_width, layout_long)
            blue_color = (255, 0, 0)  # Blue color in BGR format
            thickness_line = 5
            img_s = cv2.rectangle(blank_image, start_point_line, end_point_line, blue_color, thickness_line)
            image_saved = cv2.imwrite('greetings.png', img_s)
            text = f"LIMIT REACHED"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (0, 0, 0)  # Black color in BGR format
            font_thickness = 2
            text_position = (int((layout_width/2-75)), int((layout_long/2+5)))
            layout = cv2.imread('greetings.png')
            img_s= cv2.putText(
                layout, text, text_position, font, font_scale, font_color, font_thickness
            )
            image_saved = cv2.imwrite(f'greetings.png', img_s)
            afx = pd.DataFrame({'thislist':[f'greetings.png']})
            dfx = pd.read_csv("Images_save_1.csv")  
            df2x = pd.concat([afx,dfx], ignore_index=True) 
            df2x.to_csv('Images_save_1.csv',index = False,header=True)
            cv2.imshow("Image with Drawing", layout)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return print("LIMIT REACHED")
        else:
            collide_I_w = collide_I_w
            collide_II_l = collide_II_l

        if collide_I_l < 0:
            collide_I_l = 999999
        else:
            collide_I_l = collide_I_l

        if collide_II_l < 0:
            collide_II_l = 999999
        else:
            collide_II_l = collide_II_l
        
        if ((collide_I_w + width_p) > layout_width): 
            collide_I_w = 999999
        else:
            collide_I_w = collide_I_w

        if collide_II_w > layout_width:
            collide_II_w = 999999
        else:
            collide_II_w = collide_II_w

        dataframe = pd.read_csv("GUI_part4_distance.csv")
        pos_long_p_x = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==1]['pos_long_p'].max()
        pos_width_p_x = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==1]['pos_width_p'].max()
        
        if width_p < long_p:
            
            if (pos_width_p_x < collide_I_w) & (pos_long_p_x == collide_I_l):
                result_distance_I = abs(int(collide_I_w+layout_long-collide_I_l+(long_p)/2+looseness_corridor_l))
            else:        
                result_distance_I = abs(int(collide_I_w+layout_long-collide_I_l-(long_p)/2))

            if (pos_width_p_x < collide_II_w) & (pos_long_p_x == collide_II_l):
                result_distance_II = abs(int(collide_II_w+layout_long-collide_II_l+(long_p)/2+looseness_corridor_l))
            else:        
                result_distance_II = abs(int(collide_II_w+layout_long-collide_II_l-(long_p)/2))


        else:

            if (pos_width_p_x < collide_I_w) & (pos_long_p_x == collide_I_l):
                result_distance_I = abs(int(collide_I_w+layout_long-collide_I_l+(width_p)/2+looseness_corridor_l))
            else:        
                result_distance_I = abs(int(collide_I_w+layout_long-collide_I_l-long_p+(width_p)/2))

            if (pos_width_p_x < collide_II_w) & (pos_long_p_x == collide_II_l):
                result_distance_II = abs(int(collide_II_w+layout_long-collide_II_l+(width_p)/2+looseness_corridor_l))
            else:        
                result_distance_II = abs(int(collide_II_w+layout_long-collide_II_l-long_p+(width_p)/2))

        
        if result_distance_I < result_distance_II:
            min_collide_result = "Type_I"
        else:
            min_collide_result = "Type_II"

        if result_distance_I < result_distance_II:
            min_distance_w = collide_I_w
        else:
            min_distance_w = collide_II_w

        if result_distance_I < result_distance_II:
            min_distance_l = collide_I_l
        else:
            min_distance_l = collide_II_l

        if result_distance_I < result_distance_II:
            remains_w = collide_II_w
        else:
            remains_w = collide_I_w
        
        if result_distance_I < result_distance_II:
            remains_l = collide_II_l
        else:
            remains_l = collide_I_l

        if result_distance_I < result_distance_II:
            result_distance = result_distance_II
        else:
            result_distance = result_distance_I

        
        af = pd.DataFrame({'layout_num':[layout_num],'remains_w':[remains_w],'remains_l':[remains_l],'result_distance':[result_distance]})
        df = pd.read_csv("GUI_part4_remains.csv")  
        df2 = pd.concat([df,af], ignore_index=True)  
        df2.to_csv('GUI_part4_remains.csv',index = False,header=True)

        dataframe = pd.read_csv("GUI_part4_remains.csv") 
        remains_def_w = dataframe['remains_w'].values[0]
        remains_def_l = dataframe['remains_l'].values[0]
        remains_result_distance = dataframe['result_distance'].values[0]    
        
        pos_width_p = min_distance_w
        pos_long_p = min_distance_l
        final_pos_width_p = pos_width_p + width_p
        final_pos_long_p = pos_long_p + long_p

        af = pd.DataFrame({'layout_num':[layout_num],'ID':[ID],'layout_width':[layout_width],'layout_long':[layout_long],'width_p':[width_p],'long_p':[long_p],'looseness_corridor_w':[looseness_corridor_w],'looseness_corridor_l':[looseness_corridor_l],'collide_I_w':[collide_I_w],'collide_I_l':[collide_I_l],'collide_II_w':[collide_II_w],'collide_II_l':[collide_II_l],'result_distance_I':[result_distance_I],'result_distance_II':[result_distance_II],'min_collide_result':[min_collide_result],'min_distance_w':[min_distance_w],'min_distance_l':[min_distance_l],'remains_def_w':[remains_def_w],'remains_def_l':[remains_def_l],'remains_result_distance':[remains_result_distance],'pos_width_p':[pos_width_p],'pos_long_p':[pos_long_p],'final_pos_width_p':[final_pos_width_p],'final_pos_long_p':[final_pos_long_p],'ID_Total':[ID_Total]})
        df = pd.read_csv("GUI_part4_distance.csv")  
        df2 = pd.concat([af,df], ignore_index=True) 
        df2.to_csv('GUI_part4_distance.csv',index = False,header=True)

        start_point_rect = (pos_width_p, pos_long_p)
        end_point_rect = (final_pos_width_p, final_pos_long_p)
        red_color = (0, 0, 255)  # Red color in BGR format
        thickness_rect = 2
        layout = cv2.imread(f'Layer_{layout_num}.png')
        img_s= cv2.rectangle(layout, start_point_rect, end_point_rect, red_color, thickness_rect)
        image_saved = cv2.imwrite(f'Layer_{layout_num}.png', img_s)
        # Add text to the Area
        text = f"Area {ID_Total}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 0, 0)  # Black color in BGR format
        font_thickness = 2
        text_position = (int((pos_width_p+width_p/2-30)), int((pos_long_p+long_p/2+5)))
        layout = cv2.imread(f'Layer_{layout_num}.png')
        img_s= cv2.putText(
            layout, text, text_position, font, font_scale, font_color, font_thickness
        )
        image_saved = cv2.imwrite(f'Layer_{layout_num}.png', img_s)
        cv2.imshow("Image with Drawing", layout)
        cv2.setWindowTitle("Image with Drawing",f'Layer {layout_num}')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:

        dataframe = pd.read_csv("GUI_part4_distance.csv") 
        ID = dataframe[dataframe['layout_num']==layout_num]['ID'].max()
        
        ID = 1 + ID

        dataframe = pd.read_csv("GUI_part4_distance.csv") 
        collide_I_w = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['collide_I_w'].max()
        collide_I_l = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['collide_I_l'].max()
        collide_II_w = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['collide_II_w'].max()
        collide_II_l = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['collide_II_l'].max()
        result_distance_I = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['result_distance_I'].max()
        result_distance_II = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['result_distance_II'].max()
        min_collide_result = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['min_collide_result'].max()
        min_distance_w = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['min_distance_w'].max()
        min_distance_l = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['min_distance_l'].max()
        remains_def_w = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['remains_def_w'].max()
        remains_def_l = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['remains_def_l'].max()
        remains_result_distance = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['remains_result_distance'].max()
        pos_width_p = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['pos_width_p'].max()
        pos_long_p = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['pos_long_p'].max()
        final_pos_width_p = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['final_pos_width_p'].max()
        final_pos_long_p = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==(ID-1)]['final_pos_long_p'].max()
        
        if width_p < long_p:
                
            if min_collide_result == "Type_I":
                collide_I_w = final_pos_width_p + looseness_corridor_w
            else:   
                collide_I_w = final_pos_width_p + looseness_corridor_w

            if min_collide_result == "Type_I":
                collide_I_l = pos_long_p 
            else:   
                collide_I_l = pos_long_p 

            if min_collide_result == "Type_II":
                collide_II_w = pos_width_p
            else:   
                collide_II_w = pos_width_p

            if min_collide_result == "Type_II":
                collide_II_l = pos_long_p - looseness_corridor_l - long_p
            else:   
                collide_II_l = pos_long_p - looseness_corridor_l - long_p

        else:        

            if min_collide_result == "Type_I":
                collide_I_w = final_pos_width_p + looseness_corridor_w
            else:   
                collide_I_w = final_pos_width_p + looseness_corridor_w

            if min_collide_result == "Type_I":
                collide_I_l = pos_long_p 
            else:   
                collide_I_l = pos_long_p 

            if min_collide_result == "Type_II":
                collide_II_w = pos_width_p
            else:   
                collide_II_w = pos_width_p

            if min_collide_result == "Type_II":
                collide_II_l = pos_long_p - looseness_corridor_l - long_p
            else:   
                collide_II_l = pos_long_p - looseness_corridor_l - long_p


        if (((collide_I_w + width_p) > layout_width) and (collide_II_l<0)):
            blank_image = np.zeros((layout_long, layout_width, 3), dtype=np.uint8)
            blank_image[:] = (64, 64, 64)  # White color in BGR format
            # Layout limit
            start_point_line = (0, 0)
            end_point_line = (layout_width, layout_long)
            blue_color = (255, 0, 0)  # Blue color in BGR format
            thickness_line = 5
            img_s = cv2.rectangle(blank_image, start_point_line, end_point_line, blue_color, thickness_line)
            image_saved = cv2.imwrite('greetings.png', img_s)
            text = f"LIMIT REACHED"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (0, 0, 0)  # Black color in BGR format
            font_thickness = 2
            text_position = (int((layout_width/2-75)), int((layout_long/2+5)))
            layout = cv2.imread('greetings.png')
            img_s= cv2.putText(
                layout, text, text_position, font, font_scale, font_color, font_thickness
            )
            image_saved = cv2.imwrite(f'greetings.png', img_s)
            afx = pd.DataFrame({'thislist':[f'greetings.png']})
            dfx = pd.read_csv("Images_save_1.csv")  
            df2x = pd.concat([afx,dfx], ignore_index=True) 
            df2x.to_csv('Images_save_1.csv',index = False,header=True)
            cv2.imshow("Image with Drawing", layout)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return print("LIMIT REACHED")
        else:
            collide_I_w = collide_I_w
            collide_II_l = collide_II_l

        if collide_I_l < 0:
            collide_I_l = 999999
        else:
            collide_I_l = collide_I_l

        if collide_II_l < 0:
            collide_II_l = 999999
        else:
            collide_II_l = collide_II_l
        
        if ((collide_I_w + width_p) > layout_width): 
            collide_I_w = 999999
        else:
            collide_I_w = collide_I_w

        if collide_II_w > layout_width:
            collide_II_w = 999999
        else:
            collide_II_w = collide_II_w


        dataframe = pd.read_csv("GUI_part4_distance.csv")
        try:
            ID_y = dataframe[dataframe['layout_num']==layout_num][dataframe['pos_width_p']==collide_I_w][dataframe['pos_long_p']==collide_I_l]['ID'].max()
            pos_width_p_y = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==ID_y]['pos_width_p'].max()
            pos_long_p_y = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==ID_y]['pos_long_p'].max()
        except IndexError:
            pass
        
        if (pos_width_p_y == collide_I_w) and (pos_long_p_y == collide_I_l):
            collide_I_l = 999999
        else:
            pass

        try:
            ID_y = dataframe[dataframe['layout_num']==layout_num][dataframe['pos_width_p']==collide_II_w][dataframe['pos_long_p']==collide_II_l]['ID'].max()
            pos_width_p_y = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==ID_y]['pos_width_p'].max()
            pos_long_p_y = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==ID_y]['pos_long_p'].max()
        except IndexError:
            pass

        if(pos_width_p_y == collide_II_w) and (pos_long_p_y == collide_II_l):
            collide_II_l = 999999
        else:
            pass
    
        if ((remains_def_w == collide_I_w) and (remains_def_l == collide_I_l)) or ((remains_def_w == collide_II_w) and (remains_def_l == collide_II_l)):
            df = pd.read_csv("GUI_part4_remains.csv")  
            df2 = df[(df.layout_num != layout_num) | (df.remains_w != remains_def_w) | (df.remains_l != remains_def_l)]
            df2.to_csv('GUI_part4_remains.csv',index = False,header=True)

            dataframe = pd.read_csv("GUI_part4_remains.csv") 

            try:
                remains_result_distance = dataframe[dataframe['layout_num']==layout_num]['result_distance'].min()
                remains_def_w = dataframe[dataframe['layout_num']==layout_num][dataframe['result_distance']==remains_result_distance]['remains_w'].min()
                remains_def_l = dataframe[dataframe['layout_num']==layout_num][dataframe['remains_w']==remains_def_w]['remains_l'].min()
            except IndexError:
                pass
            
        else:
            remains_def_w = remains_def_w
            remains_def_l = remains_def_l
            remains_result_distance = remains_result_distance


        dataframe = pd.read_csv("GUI_part4_distance.csv") 
        try:
            ID_y = dataframe[dataframe['layout_num']==layout_num][dataframe['pos_width_p']==remains_def_w][dataframe['pos_long_p']==remains_def_l]['ID'].max()
            pos_width_p_y = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==ID_y]['pos_width_p'].max()
            pos_long_p_y = dataframe[dataframe['layout_num']==layout_num][dataframe['ID']==ID_y]['pos_long_p'].max()
        except IndexError:
            pass

        if (pos_width_p_y == remains_def_w) and (pos_long_p_y == remains_def_l):
            remains_def_l = 999999
        else:
            pass     

        
        dataframe = pd.read_csv("GUI_part4_distance.csv")
        pos_long_p_x = dataframe[dataframe['ID']==1]['pos_long_p'].max()
        pos_width_p_x = dataframe[dataframe['ID']==1]['pos_width_p'].max()

        if width_p < long_p:
            
            if (pos_width_p_x < collide_I_w) & (pos_long_p_x == collide_I_l):
                 result_distance_I = abs(int(collide_I_w+layout_long-collide_I_l+(long_p)/2+looseness_corridor_l))
            else:        
                result_distance_I = abs(int(collide_I_w+layout_long-collide_I_l-(long_p)/2))

            if (pos_width_p_x < collide_II_w) & (pos_long_p_x == collide_II_l):
                result_distance_II = abs(int(collide_II_w+layout_long-collide_II_l+(long_p)/2+looseness_corridor_l))
            else:        
                result_distance_II = abs(int(collide_II_w+layout_long-collide_II_l-(long_p)/2))


        else:

            if (pos_width_p_x < collide_I_w) & (pos_long_p_x == collide_I_l):
                result_distance_I = abs(int(collide_I_w+layout_long-collide_I_l+(width_p)/2+looseness_corridor_l))
            else:        
                result_distance_I = abs(int(collide_I_w+layout_long-collide_I_l-long_p+(width_p)/2))

            if (pos_width_p_x < collide_II_w) & (pos_long_p_x == collide_II_l):
                result_distance_II = abs(int(collide_II_w+layout_long-collide_II_l+(width_p)/2+looseness_corridor_l))
            else:        
                result_distance_II = abs(int(collide_II_w+layout_long-collide_II_l-long_p+(width_p)/2))
    

        if result_distance_I <= result_distance_II:
            min_collide_result = "Type_I"
        else:
            min_collide_result = "Type_II"

        if (result_distance_I <= result_distance_II) and (result_distance_I < remains_result_distance):
            min_distance_w = collide_I_w
        elif (result_distance_II < result_distance_I) and (result_distance_II < remains_result_distance):
            min_distance_w = collide_II_w
        elif (remains_result_distance <= result_distance_I) and (remains_result_distance <= result_distance_II):
            min_distance_w = remains_def_w 
        
        
        if (result_distance_I <= result_distance_II) and (result_distance_I < remains_result_distance):
            min_distance_l = collide_I_l
        elif (result_distance_II < result_distance_I) and (result_distance_II < remains_result_distance):
            min_distance_l = collide_II_l
        elif (remains_result_distance <= result_distance_I) and (remains_result_distance <= result_distance_II):
            min_distance_l = remains_def_l
        

        if ((min_distance_w + width_p) > layout_width) or (min_distance_l < 0): 
            blank_image = np.zeros((layout_long, layout_width, 3), dtype=np.uint8)
            blank_image[:] = (64, 64, 64)  # White color in BGR format
            # Layout limit
            start_point_line = (0, 0)
            end_point_line = (layout_width, layout_long)
            blue_color = (255, 0, 0)  # Blue color in BGR format
            thickness_line = 5
            img_s = cv2.rectangle(blank_image, start_point_line, end_point_line, blue_color, thickness_line)
            image_saved = cv2.imwrite('greetings.png', img_s)
            text = f"LIMIT REACHED"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (0, 0, 0)  # Black color in BGR format
            font_thickness = 2
            text_position = (int((layout_width/2-75)), int((layout_long/2+5)))
            layout = cv2.imread('greetings.png')
            img_s= cv2.putText(
                layout, text, text_position, font, font_scale, font_color, font_thickness
            )
            image_saved = cv2.imwrite(f'greetings.png', img_s)
            afx = pd.DataFrame({'thislist':[f'greetings.png']})
            dfx = pd.read_csv("Images_save_1.csv")  
            df2x = pd.concat([afx,dfx], ignore_index=True) 
            df2x.to_csv('Images_save_1.csv',index = False,header=True)
            cv2.imshow("Image with Drawing", layout)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return print("LIMIT REACHED")
        else:
            pass

        if (remains_result_distance <= result_distance_I) and (remains_result_distance <= result_distance_II):      
            af = pd.DataFrame({'layout_num':[layout_num,layout_num],'remains_w':[collide_I_w, collide_II_w],'remains_l':[collide_I_l, collide_II_l],'result_distance':[result_distance_I, result_distance_II]})
            df = pd.read_csv("GUI_part4_remains.csv")  
            df2 = pd.concat([df,af], ignore_index=True) 
            df3 = df2[(df2.layout_num != layout_num) | (df2.remains_w != remains_def_w) | (df2.remains_l != remains_def_l) | (df2.result_distance != remains_result_distance)]
            df3.to_csv('GUI_part4_remains.csv',index = False,header=True)
            df2 = pd.read_csv("GUI_part4_remains.csv")  
            df2 = df2.sort_values(by='result_distance', ascending=True)
            df2.to_csv('GUI_part4_remains.csv',index = False,header=True)
            df2 = pd.read_csv("GUI_part4_remains.csv")  
            df2 = df2.drop_duplicates()
            df2.to_csv('GUI_part4_remains.csv',index = False,header=True)

        elif (result_distance_I <= result_distance_II) and (result_distance_I < remains_result_distance):
            af = pd.DataFrame({'layout_num':[layout_num],'remains_w':[collide_II_w],'remains_l':[collide_II_l],'result_distance':[result_distance_II]})
            df = pd.read_csv("GUI_part4_remains.csv")  
            df2 = pd.concat([df,af], ignore_index=True) 
            df2.to_csv('GUI_part4_remains.csv',index = False,header=True)
            df2 = pd.read_csv("GUI_part4_remains.csv")  
            df2 = df2.sort_values(by='result_distance', ascending=True)
            df2.to_csv('GUI_part4_remains.csv',index = False,header=True)
            df2 = pd.read_csv("GUI_part4_remains.csv")  
            df2 = df2.drop_duplicates()
            df2.to_csv('GUI_part4_remains.csv',index = False,header=True)
        
        elif (result_distance_II < result_distance_I) and (result_distance_II < remains_result_distance):
            af = pd.DataFrame({'layout_num':[layout_num],'remains_w':[collide_I_w],'remains_l':[collide_I_l],'result_distance':[result_distance_I]})
            df = pd.read_csv("GUI_part4_remains.csv")  
            df2 = pd.concat([df,af], ignore_index=True) 
            df2.to_csv('GUI_part4_remains.csv',index = False,header=True)
            df2 = pd.read_csv("GUI_part4_remains.csv")  
            df2 = df2.sort_values(by='result_distance', ascending=True)
            df2.to_csv('GUI_part4_remains.csv',index = False,header=True)
            df2 = pd.read_csv("GUI_part4_remains.csv")  
            df2 = df2.drop_duplicates()
            df2.to_csv('GUI_part4_remains.csv',index = False,header=True)
        
        
        dataframe = pd.read_csv("GUI_part4_remains.csv") 
        remains_result_distance = dataframe[dataframe['layout_num']==layout_num]['result_distance'].min()
        remains_def_w = dataframe[dataframe['layout_num']==layout_num][dataframe['result_distance']==remains_result_distance]['remains_w'].min()
        remains_def_l = dataframe[dataframe['layout_num']==layout_num][dataframe['result_distance']==remains_result_distance][dataframe['remains_w']==remains_def_w]['remains_l'].min()
        
        
        pos_width_p = min_distance_w
        pos_long_p = min_distance_l
        final_pos_width_p = pos_width_p + width_p
        final_pos_long_p = pos_long_p + long_p 

        af = pd.DataFrame({'layout_num':[layout_num],'ID':[ID],'layout_width':[layout_width],'layout_long':[layout_long],'width_p':[width_p],'long_p':[long_p],'looseness_corridor_w':[looseness_corridor_w],'looseness_corridor_l':[looseness_corridor_l],'collide_I_w':[collide_I_w],'collide_I_l':[collide_I_l],'collide_II_w':[collide_II_w],'collide_II_l':[collide_II_l],'result_distance_I':[result_distance_I],'result_distance_II':[result_distance_II],'min_collide_result':[min_collide_result],'min_distance_w':[min_distance_w],'min_distance_l':[min_distance_l],'remains_def_w':[remains_def_w],'remains_def_l':[remains_def_l],'remains_result_distance':[remains_result_distance],'pos_width_p':[pos_width_p],'pos_long_p':[pos_long_p],'final_pos_width_p':[final_pos_width_p],'final_pos_long_p':[final_pos_long_p],'ID_Total':[ID_Total]})
        df = pd.read_csv("GUI_part4_distance.csv")  
        df2 = pd.concat([af,df], ignore_index=True) 
        df2.to_csv('GUI_part4_distance.csv',index = False,header=True)

        start_point_rect = (pos_width_p, pos_long_p)
        end_point_rect = (final_pos_width_p, final_pos_long_p)
        red_color = (0, 0, 255)  # Red color in BGR format
        thickness_rect = 2
        layout = cv2.imread(f'Layer_{layout_num}.png')
        img_s= cv2.rectangle(layout, start_point_rect, end_point_rect, red_color, thickness_rect)
        image_saved = cv2.imwrite(f'Layer_{layout_num}.png', img_s)
        # Add text 
        # to the Area
        text = f"Area {ID_Total}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 0, 0)  # Black color in BGR format
        font_thickness = 2
        text_position = (int((pos_width_p+width_p/2-30)), int((pos_long_p+long_p/2+5)))
        layout = cv2.imread(f'Layer_{layout_num}.png')
        img_s= cv2.putText(
            layout, text, text_position, font, font_scale, font_color, font_thickness
        )
        image_saved = cv2.imwrite(f'Layer_{layout_num}.png', img_s)
        cv2.imshow("Image with Drawing", layout)
        cv2.setWindowTitle("Image with Drawing",f'Layer {layout_num}')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

button = customtkinter.CTkButton(widgets_frame, text="Insert Area", command=area)
button.grid(row=11, column=0, padx=5, pady=5, sticky="nsew")

import os
from tkinter.filedialog import askopenfilename 
from tkinter import filedialog

from subprocess import call

def open_py_file(): 
    
    import zipfile
    import os
    
    dfx = pd.read_csv("Images_save_1.csv")  
    thislist = dfx['thislist'].tolist()    
    thislist = list(set(thislist))

    # Create a zip file
    with zipfile.ZipFile('Normal_Multilayer_Areas.zip', 'w') as zip_file:
        # Add each image to the zip file
        for image_path in thislist:
            # Add the image file to the zip file with its original name
            zip_file.write(image_path, os.path.basename(image_path))

    thislist.append('Normal_Multilayer_Areas.zip')

    call(["python","1stconcept_part15.py"])
    

button = customtkinter.CTkButton(widgets_frame, text="Product Allocation", command=open_py_file)
button.grid(row=12, column=0, padx=5, pady=5, sticky="nsew")


def save(): 
    
    from datetime import datetime
    u={'save_as_type':["Normal Multilayer"], 'save_as_time':[datetime.now()]}
    dataframe = pd.DataFrame(u)
    dataframe.to_csv("save_as_type.csv",index = False,header=True)

    import zipfile
    import os
    
    dfx = pd.read_csv("Images_save_1.csv")  
    thislist = dfx['thislist'].tolist()
    thislist = list(set(thislist))
    
    # Create a zip file
    with zipfile.ZipFile('Normal_Multilayer_Areas.zip', 'w') as zip_file:
        # Add each image to the zip file
        for image_path in thislist:
            # Add the image file to the zip file with its original name
            zip_file.write(image_path, os.path.basename(image_path))

    thislist.append('Normal_Multilayer_Areas.zip')

    call(["python","Save.py"])

button_2 = customtkinter.CTkButton(widgets_frame, text="Save", command=save)
button_2.grid(row=13, column=0, padx=5, pady=5, sticky="nsew")

def exit(): 
    
    try:

        dfx = pd.read_csv("Images_save_1.csv")  
        thislist = dfx['thislist'].tolist()
        thislist = list(set(thislist))
        for image_path in thislist:
            os.remove(image_path)
        
        dfx_2 = pd.read_csv("Images_save.csv")  
        thislist_2 = dfx_2['thislist'].tolist()
        thislist_2 = list(set(thislist_2))
        for image_path in thislist_2:
            os.remove(image_path)
    
    except FileNotFoundError:
        pass


button_3 = customtkinter.CTkButton(widgets_frame, text="Exit", command=lambda: [exit(), root.destroy()])
button_3.grid(row=14, column=0, padx=5, pady=5, sticky="nsew")

root.mainloop()    