import pandas as pd
import pickle

t = {'Priority':[0],'Perishable':[0],'Fragile':[0],'Weight':[0],'Floor':[0]}
dataframe_1 = pd.DataFrame(t)
dataframe_1.to_csv("GUI_AI_2.csv",index = False)


import tkinter as tk
from tkinter import ttk, PhotoImage
import cv2
import numpy as np
import pandas as pd
import psutil

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import customtkinter

from datetime import datetime, timedelta

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

    u = {'layout_num':[],'Area':[],'Area_width':[],'Layout_width':[],'Layout_height':[],'floor_width':[],'floor_height':[],'floor_1_w_i':[],'floor_1_h_i':[],'floor_1_w_f':[],'floor_1_h_f':[],'Floor_2_w_i':[],'Floor_2_h_i':[],'Floor_2_w_f':[],'Floor_2_h_f':[],'Floor_3_w_i':[],'Floor_3_h_i':[],'Floor_3_w_f':[],'Floor_3_h_f':[],'Area_category':[]}
    dataframe_2 = pd.DataFrame(u)
    dataframe_2.to_csv("GUI_area_Database.csv",index = False)

    d = {'layout_num':[],'Area':[],'Area_width':[],'Layout_width':[],'Layout_height':[],'floor_width':[],'floor_height':[],'floor_1_w_i':[],'floor_1_h_i':[],'floor_1_w_f':[],'floor_1_h_f':[],'Floor_2_w_i':[],'Floor_2_h_i':[],'Floor_2_w_f':[],'Floor_2_h_f':[],'Floor_3_w_i':[],'Floor_3_h_i':[],'Floor_3_w_f':[],'Floor_3_h_f':[],'p_width':[],'p_height':[],'p_weight':[],'p_floor':[],'p_floor_1_w':[],'p_floor_2_w':[],'p_floor_3_w':[],'p_width_i':[],'p_height_i':[],'p_width_f':[],'p_height_f':[],'Number':[],'p_code':[],'Area_category':[],'Company_Brand':[]}
    dataframe = pd.DataFrame(d)
    dataframe.to_csv("GUI_product_Database.csv",index = False)

    u = {'thislist':[]}
    dataframe_u = pd.DataFrame(u)
    dataframe_u.to_csv("Images_save.csv",index = False)


root = customtkinter.CTk()

frame = customtkinter.CTkFrame(master=root)
frame.pack()

widgets_frame = customtkinter.CTkLabel(frame, text="Warehouse Allocation Process")
widgets_frame.grid(row=0, column=0, padx=35, pady=22)

widgets_frame = customtkinter.CTkLabel(frame, text="Company's Brand")
widgets_frame.grid(row=1, column=0, padx=5, pady=5)

combo_list_1 = ["Shunchen Sports Jia", "NIKE", "361 Du KIDS", "Coca-Cola", "PepsiCo", "Pringles", "Zara", "H&M", "Uniqlo", "Gap"]

def combobox_callback(choice):
    print("combobox dropdown clicked:", choice)

status_combobox_1 = customtkinter.CTkComboBox(master=widgets_frame,
                                     values=combo_list_1,
                                     command=combobox_callback)
status_combobox_1.grid(row=2, column=0, padx=5, pady=5,  sticky="ew")
status_combobox_1.set(combo_list_1[0])

Width = customtkinter.CTkEntry(widgets_frame)
Width.insert(0, "Width of product")
Width.bind("<FocusIn>", lambda e: Width.delete('0', 'end'))
Width.grid(row=3, column=0, padx=5, pady=(0, 5), sticky="ew")

Long = customtkinter.CTkEntry(widgets_frame)
Long.insert(0, "Height of product")
#Long.bind("<FocusIn>", lambda e: Long.delete('0', 'end'))
Long.grid(row=4, column=0, padx=5, pady=(0, 5), sticky="ew")  

Weight_1 = customtkinter.CTkEntry(widgets_frame)
Weight_1.insert(0, "Weight of product")
#Weight.bind("<FocusIn>", lambda e: Long.delete('0', 'end'))
Weight_1.grid(row=5, column=0, padx=5, pady=(0, 5), sticky="ew")  


b = tk.BooleanVar()
checkbutton =  customtkinter.CTkCheckBox(widgets_frame, text="Priority", variable=b)
checkbutton.grid(row=6, column=0, padx=5, pady=5, sticky="nsew")


a = tk.BooleanVar()
checkbutton =  customtkinter.CTkCheckBox(widgets_frame, text="Fragile", variable=a)
checkbutton.grid(row=7, column=0, padx=5, pady=5, sticky="nsew")


def area_product_number():


    combobox_3 = "Priority" if b.get() else "Not Priority"
    if combobox_3 == "Priority":
        combobox_3 = int(1)
    elif combobox_3 == "Not Priority":
        combobox_3 = int(0)

    combobox_1 = status_combobox_1.get()
    if (combobox_1 == 'Coca-Cola') or (combobox_1 == 'PepsiCo') or (combobox_1 == 'Pringles'): 
        combobox_5 = int(1)
    else:
        combobox_5 = int(0)

    combobox_6 = "Fragile" if a.get() else "Not Fragile"
    if combobox_6 == "Fragile":
        combobox_6 = int(1)
    elif combobox_6 == "Not Fragile":
        combobox_6 = int(0)
    
    weight = int(Weight_1.get())

    status = status_combobox_2.get()


    # Convert all GUI inputs to a feature dictionary
    product_features = {
        
        'Priority': 1 if b.get() else 0,  # Simplified priority conversion
        'Perishable': 1 if status_combobox_1.get() in ['Coca-Cola', 'PepsiCo', 'Pringles'] else 0,
        'Fragile': 1 if a.get() else 0,  # Simplified fragility conversion
        'Weight': int(Weight_1.get()),  # From your weight entry field
        # Add other features here matching your training data columns
        # Example: 'Height': int(height_entry.get()),
        #          'Width': int(width_entry.get())
    }
        
    from datetime import datetime, timedelta
    dataframe = pd.read_csv("floor_config.csv")

    #Limit weight per floor
    K_levels = dataframe['levels'].values[0]
    weight_choice = dataframe['weight_choice'].values[0]

    f = open('warehouse_level_predictor.pkl', 'rb')
    models = pickle.load(f)

    # Convert input to DataFrame with correct feature order
    X_new = pd.DataFrame([product_features])[models['feature_columns']]

    # Scale and predict
    X_new_scaled = models['scaler'].transform(X_new)
    p_floor = models['kmeans'].predict(X_new_scaled)[0] + 1


    width = int(Width.get())
    long = int(Long.get())
    weight = int(Weight_1.get())

    p_width = width    
    p_height = long
    p_weight = weight 

    Floor = p_floor  

    combobox_1 = status_combobox_1.get() 
    
    # For Area (Number of Shelf at the beginning of the allocation process for each brand of product)
    
    from datetime import datetime

    df = pd.read_csv("Static_Training_Dataset_2.csv",sep = ',')

    df['Time_start'] = pd.to_datetime(df['Time_start'])
    df['Time_finish'] = pd.to_datetime(df['Time_finish'])

    df['Time_spent'] = df['Time_finish'] - df['Time_start'] 

    df_2= df.groupby("Company_Brand")['Time_spent'].sum()

    df_2 = pd.DataFrame({'Company_Brand':df_2.index, 'Sum_Time':df_2.values})

    df_2 = df_2.sort_values(by='Sum_Time', ascending=True)
    Total_Sum_Time = df_2['Sum_Time'].sum()
    df_2['%Time'] = df_2['Sum_Time']/Total_Sum_Time
    df_2['%Acum_Time'] = df_2['%Time'].cumsum(axis=0)
    df_2.loc[df_2['%Acum_Time'] <= 0.1, 'Pareto_Time'] = 'A' 
    df_2.loc[(df_2['%Acum_Time'] <= 0.25) & (df_2['%Acum_Time'] > 0.1), 'Pareto_Time'] = 'B'
    df_2.loc[(df_2['%Acum_Time'] <= 1) & (df_2['%Acum_Time'] > 0.25), 'Pareto_Time'] = 'C' 

    df_2.loc[df_2['Pareto_Time'] == 'A', 'Pareto_Time_num'] = 3
    df_2.loc[df_2['Pareto_Time'] == 'B', 'Pareto_Time_num'] = 2
    df_2.loc[df_2['Pareto_Time'] == 'C', 'Pareto_Time_num'] = 1

    df_2.to_csv('Pareto_Time.csv',index = False,header=True)

    df_2_y = pd.read_csv("Pareto_Time.csv")
    s = {}
    df_2_x = pd.DataFrame(s)
    df_2_x['Time_num'] = range(1, 1 + len(df_2_y))
    df_2_x = df_2_x.sort_values(by='Time_num', ascending=False)
    df_2_x = df_2_x.reset_index()
    df_2_y['Time_num'] = df_2_x['Time_num'] 

    df_3= df.groupby("Company_Brand")['Company_Brand'].count()

    df_3 = pd.DataFrame({'Company_Brand':df_3.index, 'Count':df_3.values})

    df_3 = df_3.sort_values(by='Count', ascending=False)
    Total_Count = df_3['Count'].sum()
    df_3['%_Frequency'] = df_3['Count']/Total_Count
    df_3['%Acum_Frequency'] = df_3['%_Frequency'].cumsum(axis=0)
    df_3.loc[df_3['%Acum_Frequency'] <= 0.75, 'Pareto_Frequency'] = 'A' 
    df_3.loc[(df_3['%Acum_Frequency'] <= 0.90) & (df_3['%Acum_Frequency'] > 0.75), 'Pareto_Frequency'] = 'B'
    df_3.loc[(df_3['%Acum_Frequency'] <= 1) & (df_3['%Acum_Frequency'] > 0.90), 'Pareto_Frequency'] = 'C' 

    df_3.loc[df_3['Pareto_Frequency'] == 'A', 'Pareto_Frequency_num'] = 3
    df_3.loc[df_3['Pareto_Frequency'] == 'B', 'Pareto_Frequency_num'] = 2
    df_3.loc[df_3['Pareto_Frequency'] == 'C', 'Pareto_Frequency_num'] = 1

    df_3.to_csv('Pareto_Frequency.csv',index = False,header=True)

    df_3_y = pd.read_csv("Pareto_Frequency.csv")
    s = {}
    df_3_x = pd.DataFrame(s)
    df_3_x['Frequency_num'] = range(1, 1 + len(df_3_y))
    df_3_x = df_3_x.sort_values(by='Frequency_num', ascending=False)
    df_3_x = df_3_x.reset_index()
    df_3_y['Frequency_num'] = df_3_x['Frequency_num'] 


    df_4 = pd.merge(df_2, df_3, on=['Company_Brand'])
    df_4 = df_4[['Company_Brand','Sum_Time','Count']]
    df_4['Time_per_spent'] = df_4['Sum_Time']/df_4['Count']
    df_4 = df_4.sort_values(by='Time_per_spent', ascending=True)
    Total_Time_per_spent = df_4['Time_per_spent'].sum()
    df_4['%Time_per_spent'] = df_4['Time_per_spent']/Total_Time_per_spent
    df_4.drop(['Sum_Time', 'Count'], axis=1)
    df_4['%Acum_Time_per_spent'] = df_4['%Time_per_spent'].cumsum(axis=0)
    df_4.loc[df_4['%Acum_Time_per_spent'] <= 0.10, 'Pareto_Time_per_spent'] = 'A' 
    df_4.loc[(df_4['%Acum_Time_per_spent'] <= 0.25) & (df_4['%Acum_Time_per_spent'] > 0.10), 'Pareto_Time_per_spent'] = 'B'
    df_4.loc[(df_4['%Acum_Time_per_spent'] <= 1) & (df_4['%Acum_Time_per_spent'] > 0.25), 'Pareto_Time_per_spent'] = 'C' 

    df_4.loc[df_4['Pareto_Time_per_spent'] == 'A', 'Pareto_Time_per_spent_num'] = 3
    df_4.loc[df_4['Pareto_Time_per_spent'] == 'B', 'Pareto_Time_per_spent_num'] = 2
    df_4.loc[df_4['Pareto_Time_per_spent'] == 'C', 'Pareto_Time_per_spent_num'] = 1

    df_4.to_csv('Pareto_Time_per_spent.csv',index = False,header=True)


    df_4_y = pd.read_csv("Pareto_Time_per_spent.csv")
    s = {}
    df_4_x = pd.DataFrame(s)
    df_4_x['Time_per_spent_num'] = range(1, 1 + len(df_4_y))
    df_4_x = df_4_x.sort_values(by='Time_per_spent_num', ascending=False)
    df_4_x = df_4_x.reset_index()
    df_4_y['Time_per_spent_num'] = df_4_x['Time_per_spent_num'] 

    df_5 = df.groupby("Company_Brand")['Income'].sum()

    df_5 = pd.DataFrame({'Company_Brand':df_5.index, 'Sum_Prof':df_5.values})

    df_5 = df_5.sort_values(by='Sum_Prof', ascending=False)
    Total_Prof = df_5['Sum_Prof'].sum()
    df_5['%Profitable'] = df_5['Sum_Prof']/Total_Prof
    df_5['%Acum_Profitable'] = df_5['%Profitable'].cumsum(axis=0)
    df_5.loc[df_5['%Acum_Profitable'] <= 0.75, 'Pareto_Profitability'] = 'A' 
    df_5.loc[(df_5['%Acum_Profitable'] <= 0.90) & (df_5['%Acum_Profitable'] > 0.75), 'Pareto_Profitability'] = 'B'
    df_5.loc[(df_5['%Acum_Profitable'] <= 1) & (df_5['%Acum_Profitable'] > 0.90), 'Pareto_Profitability'] = 'C' 

    df_5.loc[df_5['Pareto_Profitability'] == 'A', 'Pareto_Profitability_num'] = 3
    df_5.loc[df_5['Pareto_Profitability'] == 'B', 'Pareto_Profitability_num'] = 2
    df_5.loc[df_5['Pareto_Profitability'] == 'C', 'Pareto_Profitability_num'] = 1

    df_5.to_csv('Pareto_Profitability.csv',index = False,header=True)

    df_5_y = pd.read_csv("Pareto_Profitability.csv")
    s = {}
    df_5_x = pd.DataFrame(s)
    df_5_x['Profitability_num'] = range(1, 1 + len(df_5_y))
    df_5_x = df_5_x.sort_values(by='Profitability_num', ascending=False)
    df_5_x = df_5_x.reset_index()
    df_5_y['Profitability_num'] = df_5_x['Profitability_num'] 


    data = pd.read_csv("weights.csv")
    weight_time = data[data['weight_type']=='time weight']['weight'].max() 
    weight_frequency = data[data['weight_type']=='frequency weight']['weight'].max() 
    weight_time_per_spent = data[data['weight_type']=='per item time spent weight']['weight'].max() 
    weight_profitability = data[data['weight_type']=='profit weight']['weight'].max()     


    df_6 = pd.merge(df_2, df_3, on=['Company_Brand'])
    df_6 = pd.merge(df_4, df_6, on=['Company_Brand'])
    df_6 = pd.merge(df_5, df_6, on=['Company_Brand'])

    s = {}
    df_6_x = pd.DataFrame(s)
    if(weight_time > 0):
        df_6_x['Pareto_Time'] = df_6['Pareto_Time']
    else:
        df_6_x['Pareto_Time'] = None
    if(weight_frequency > 0):
        df_6_x['Pareto_Frequency'] = df_6['Pareto_Frequency']
    else:
        df_6_x['Pareto_Frequency'] = None
    if(weight_time_per_spent > 0):
        df_6_x['Pareto_Time_per_spent'] = df_6['Pareto_Time_per_spent']
    else:
        df_6_x['Pareto_Time_per_spent'] = None
    if(weight_profitability > 0):
        df_6_x['Pareto_Profitability'] = df_6['Pareto_Profitability']
    else:
        df_6_x['Pareto_Profitability'] = None

    df_6_x.fillna('', inplace=True)

    df_6['Multi_Pareto_Classification'] = df_6_x['Pareto_Time'] + df_6_x['Pareto_Frequency'] + df_6_x['Pareto_Time_per_spent'] + df_6_x['Pareto_Profitability']

    df_6 = df_6[['Company_Brand', 'Pareto_Time_per_spent_num', 'Pareto_Profitability_num', 'Pareto_Frequency_num', 'Pareto_Time_num','Multi_Pareto_Classification']]
    df_6['Multi_Pareto_Result'] = df_6['Pareto_Time_num']*weight_time + df_6['Pareto_Frequency_num']*weight_frequency + df_6['Pareto_Time_per_spent_num']*weight_time_per_spent + df_6['Pareto_Profitability_num']*weight_profitability
    df_6 = df_6.sort_values(by='Multi_Pareto_Result', ascending=False)
    df_6.to_csv('Multi_Pareto.csv',index = False,header=True)

    df_7 = df_6[['Company_Brand', 'Multi_Pareto_Result']]
    df_7 = pd.merge(df_7, df_3, on=['Company_Brand'])
    df_7 = df_7[['Company_Brand', 'Multi_Pareto_Result', '%_Frequency']]
    dataframe = pd.read_csv("GUI_part4_distance.csv") 
    layout_num_maxi = dataframe['layout_num'].max()
    Shelfs_max = dataframe[dataframe['layout_num']==layout_num_maxi]['ID_Total'].max()
    df_7['Shelves'] = df_7['%_Frequency']*Shelfs_max
    df_7['Acum_Shelf'] = df_7['Shelves'].cumsum(axis=0)
    df_7.loc[(df_7['Acum_Shelf'].astype(int) == Shelfs_max), 'N°_of_Shelf_f_prev'] = (df_7['Acum_Shelf']).astype(int)
    df_7.loc[(df_7['Acum_Shelf'] < Shelfs_max), 'N°_of_Shelf_f_prev'] = (df_7['Acum_Shelf']).astype(int)+1

    df_8_x = df_7.groupby("Multi_Pareto_Result")['N°_of_Shelf_f_prev'].max()
    df_8_x = pd.DataFrame({'Multi_Pareto_Result':df_8_x.index, 'N°_of_Shelf_f':df_8_x.values})
    df_8_x = df_8_x.sort_values(by='Multi_Pareto_Result', ascending=False)
    df_8_x.to_csv('Multi_Pareto_Shelf_prev.csv',index = False,header=True)
    df_8 = pd.read_csv("Multi_Pareto_Shelf_prev.csv")

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice.csv",index = False,header=True)

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice_2.csv",index = False,header=True)

    first_row = df_8['N°_of_Shelf_f'].values[0]
    first_multi_pareto_result = df_8['Multi_Pareto_Result'].values[0]

    def get_shelf_i(shelf_i):
        y = -1

        dataframe = pd.read_csv("artifice.csv")
        artifice = dataframe['artifice'].values[0]

        while(y<artifice-1):
            y = y+1

        if (shelf_i == first_row):
            if(first_multi_pareto_result == df_8['Multi_Pareto_Result'].values[y]):
                dataframe = pd.read_csv("artifice.csv")
                dataframe = dataframe.drop(['artifice'], axis=1)
                dataframe.insert(0,"artifice",[artifice+1],False)
                dataframe.to_csv("artifice.csv",index = False,header=True)
                return 1
            else:
                s={'artifice':[1]}
                dataframe = pd.DataFrame(s)
                dataframe.to_csv("artifice.csv",index = False,header=True)
                return df_8['N°_of_Shelf_f'].values[y]

        else:
            i = -1

            dataframe = pd.read_csv("artifice_2.csv")
            artifice = dataframe['artifice'].values[0]
            while(i<artifice-1):
                i = i+1
                    
                dataframe = pd.read_csv("artifice_2.csv")
                dataframe = dataframe.drop(['artifice'], axis=1)
                dataframe.insert(0,"artifice",[artifice+1],False)
                dataframe.to_csv("artifice_2.csv",index = False,header=True)
            
            if (df_8['Multi_Pareto_Result'].values[i]==df_8['Multi_Pareto_Result'].values[i-1]):
                return df_8['N°_of_Shelf_f'].values[i-1]        
            else:        
                return df_8['N°_of_Shelf_f'].values[i]
          

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice.csv",index = False,header=True)

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice_2.csv",index = False,header=True)

    df_8['N°_of_Shelf_i'] = df_8['N°_of_Shelf_f'].apply(get_shelf_i)

    df_7 = pd.merge(df_7, df_8, on=['Multi_Pareto_Result'])
    df_7.to_csv('Multi_Pareto_Shelf.csv',index = False,header=True)

    df_6_y = pd.merge(df_2_y, df_3_y, on=['Company_Brand'])
    df_6_y = pd.merge(df_4_y, df_6_y, on=['Company_Brand'])
    df_6_y = pd.merge(df_5_y, df_6_y, on=['Company_Brand'])
    df_6_y = df_6_y[['Company_Brand', 'Time_per_spent_num', 'Profitability_num', 'Frequency_num', 'Time_num']]
    df_6_y['Distinct_Result'] = df_6_y['Time_num']*weight_time + df_6_y['Frequency_num']*weight_frequency + df_6_y['Time_per_spent_num']*weight_time_per_spent + df_6_y['Profitability_num']*weight_profitability
    df_6_y = df_6_y.sort_values(by='Distinct_Result', ascending=False)
    df_6_y.to_csv('Distinct_Result.csv',index = False,header=True)

    df_7_y = df_6_y[['Company_Brand', 'Distinct_Result']]
    df_7_y = pd.merge(df_7_y, df_3_y, on=['Company_Brand'])
    df_7_y = df_7_y[['Company_Brand', 'Distinct_Result', '%_Frequency']]
    dataframe = pd.read_csv("GUI_part4_distance.csv")
    layout_num = dataframe['layout_num'].max() 
    Shelfs_max = dataframe[dataframe['layout_num']==layout_num]['ID'].max()
    df_7_y['Shelves'] = df_7_y['%_Frequency']*Shelfs_max
    df_7_y['Acum_Shelf'] = df_7_y['Shelves'].cumsum(axis=0)
    df_7_y.loc[((df_7_y['Acum_Shelf']).astype(int) == Shelfs_max), 'N°_of_Shelf_f_prev'] = (df_7_y['Acum_Shelf']).astype(int)
    df_7_y.loc[(df_7_y['Acum_Shelf'] < Shelfs_max), 'N°_of_Shelf_f_prev'] = (df_7_y['Acum_Shelf']).astype(int)+1

    df_8_y_2 = df_7_y.groupby("Distinct_Result")['N°_of_Shelf_f_prev'].max()
    df_8_y_2 = pd.DataFrame({'Distinct_Result':df_8_y_2.index, 'N°_of_Shelf_f':df_8_y_2.values})
    df_8_y_2 = df_8_y_2.sort_values(by='Distinct_Result', ascending=False)
    df_8_y = df_8_y_2.reset_index()
    first_row = df_8_y['N°_of_Shelf_f'].values[0]
    first_distinct_result = df_8_y['Distinct_Result'].values[0]

    def get_shelf_i(shelf_i):
        y = -1

        dataframe = pd.read_csv("artifice.csv")
        artifice = dataframe['artifice'].values[0]

        while(y<artifice-1):
            y = y+1

        if (shelf_i == first_row) :
            if(first_distinct_result == df_8_y['Distinct_Result'].values[y]):
                dataframe = pd.read_csv("artifice.csv")
                dataframe = dataframe.drop(['artifice'], axis=1)
                dataframe.insert(0,"artifice",[artifice+1],False)
                dataframe.to_csv("artifice.csv",index = False,header=True)
                return 1
            else:
                s={'artifice':[1]}
                dataframe = pd.DataFrame(s)
                dataframe.to_csv("artifice.csv",index = False,header=True)
                return df_8_y['N°_of_Shelf_f'].values[y]

        else:
            i = -1

            dataframe = pd.read_csv("artifice_2.csv")
            artifice = dataframe['artifice'].values[0]
            
            while(i<artifice-1):
                i = i+1
                dataframe = pd.read_csv("artifice_2.csv")
                dataframe = dataframe.drop(['artifice'], axis=1)
                dataframe.insert(0,"artifice",[artifice+1],False)
                dataframe.to_csv("artifice_2.csv",index = False,header=True)
            
            data = pd.read_csv("weights.csv")
            weight_time = data[data['weight_type']=='time weight']['weight'].max() 
            weight_frequency = data[data['weight_type']=='frequency weight']['weight'].max()
            
            if(weight_time > 0)|(weight_frequency > 0):
                if (df_8_y['Distinct_Result'].values[i]==df_8_y['Distinct_Result'].values[i+1]):
                    return df_8_y['N°_of_Shelf_f'].values[i]        
                else: 
                    return df_8_y['N°_of_Shelf_f'].values[i+1]
            else:
                if (df_8_y['Distinct_Result'].values[i]==df_8_y['Distinct_Result'].values[i-1]):
                    return df_8_y['N°_of_Shelf_f'].values[i-1]        
                else: 
                    return df_8_y['N°_of_Shelf_f'].values[i]
                

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice.csv",index = False,header=True)

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice_2.csv",index = False,header=True)

    df_8_y['N°_of_Shelf_i'] = df_8_y['N°_of_Shelf_f'].apply(get_shelf_i)

    df_7_y = pd.merge(df_7_y, df_8_y, on=['Distinct_Result'])

    df_7_y = df_7_y.drop(['index'], axis=1)

    df_7_y.to_csv('Distinct_Result_Shelf.csv',index = False,header=True)

    df_7_y_2 = pd.read_csv("Distinct_Result_Shelf.csv")

    df_7_x_2 = df_7_y_2.groupby("N°_of_Shelf_i")['N°_of_Shelf_i'].max()
    df_7_x_2 = pd.DataFrame({'N°_of_Shelf_i':df_7_x_2.index, 'Shelf_classification_prev':df_7_x_2.values})
    df_7_x_2 = df_7_x_2.sort_values(by='Shelf_classification_prev', ascending=True)
    df_7_x_2 = df_7_x_2.reset_index()
    df_7_x_2['Shelf_classification'] = range(1, 1 + len(df_7_x_2))


    num_count = df_7_x_2['Shelf_classification'].count()

    def get_shelf_letter_classification(shelf_letter):
        
        i = -1

        dataframe = pd.read_csv("artifice.csv")
        artifice = dataframe['artifice'].values[0]

        while(i<artifice-1):
            i = i+1
        
        if (artifice<(num_count)):
            dataframe = pd.read_csv("artifice.csv")
            dataframe = dataframe.drop(['artifice'], axis=1)
            dataframe.insert(0,"artifice",[artifice+1],False)
            dataframe.to_csv("artifice.csv",index = False,header=True)
        
        return chr(i+65)

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice.csv",index = False,header=True)

    df_7_x_2['Shelf_classification_letter'] = df_7_x_2['Shelf_classification'].apply(get_shelf_letter_classification)

    df_7_y_2 = pd.merge(df_7_y_2, df_7_x_2, on=['N°_of_Shelf_i'])
    df_7_y_2 = df_7_y_2.drop(['index','Shelf_classification_prev','Shelf_classification'], axis=1)
    df_7_y_2.to_csv('Distinct_Result_Shelf.csv',index = False,header=True)

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice.csv",index = False,header=True)

    dff = pd.read_csv("Item_classification.csv") 
    Item_classication = dff['Item_classification'].values[0]
    
    if Item_classication == "Pareto Analysis":
        dataframe = pd.read_csv("Multi_Pareto.csv") 
        dataframe_2 = dataframe[['Company_Brand', 'Multi_Pareto_Classification']]
        dataframe_3 = pd.read_csv("Multi_Pareto_Shelf.csv") 
        dataframe_4 = pd.merge(dataframe_3, dataframe_2, on=['Company_Brand'])
        dataframe_4['Classification'] = dataframe_4['Multi_Pareto_Classification'] 
        dataframe_5 = dataframe_4[['Company_Brand', 'N°_of_Shelf_i', 'Classification']]
        dataframe_5.to_csv('Set_up_2.csv',index = False,header=True) 
        
    elif Item_classication == "Distinct Category":
        dataframe = pd.read_csv("Distinct_Result_Shelf.csv") 
        dataframe['Classification'] = dataframe['Shelf_classification_letter']
        dataframe_2 = dataframe[['Company_Brand', 'N°_of_Shelf_i', 'Classification']]
        dataframe_2.to_csv('Set_up_2.csv',index = False,header=True) 
    
    if Item_classication == "AI":
        # Load and preprocess data
        df = pd.read_csv("Static_Training_Dataset_2.csv", sep=',')
        df['Time_start'] = pd.to_datetime(df['Time_start'])
        df['Time_finish'] = pd.to_datetime(df['Time_finish'])
        df['Time_spent'] = (df['Time_finish'] - df['Time_start']).dt.total_seconds()  # Convert to seconds

        # Invert Time_spent (lower values = faster movement = higher priority)
        df['Time_priority'] = 1 / (1 + df['Time_spent'])  # Adding 1 to avoid division by zero

        # Create frequency count per brand
        brand_counts = df['Company_Brand'].value_counts().reset_index()
        brand_counts.columns = ['Company_Brand', 'Count']
        df = pd.merge(df, brand_counts, on='Company_Brand')

        # Select features for K-means
        kmeans_df = df[['Income', 'Time_priority', 'Count']].copy()

        # Standardize features (critical for K-means)
        scaler = StandardScaler()
        kmeans_data = scaler.fit_transform(kmeans_df)

        dataframe = pd.read_csv("GUI_part4_distance.csv") 
        Total_shelves = dataframe['ID_Total'].max()    

        # Apply K-means
        kmeans = KMeans(n_clusters=Total_shelves, random_state=42)
        df['Shelf_Cluster'] = kmeans.fit_predict(kmeans_data) + 1

        # Analyze cluster characteristics
        cluster_summary = df.groupby('Shelf_Cluster')[['Income', 'Time_priority', 'Count']].mean()
        print(cluster_summary)

        # Save results
        df.to_csv('KMeans_shelf_locations.csv', index=False)

        # Create a brand-cluster lookup table
        brand_cluster_map = df.groupby('Company_Brand')['Shelf_Cluster'].agg(lambda x: x.mode()[0]).to_dict()

        # print(brand_cluster_map)


        # if combobox_1 in brand_cluster_map:
        Area = brand_cluster_map[combobox_1]
        Area_category = 'K means'
        
        # Rule 2: Default cluster for unknown brands
        # else:
        #     return default_cluster  # e.g., cluster with most frequent items

        

    else:
        dataframe_clasification = pd.read_csv("Set_up_2.csv") 
        Area = int(dataframe_clasification[dataframe_clasification['Company_Brand']== combobox_1]['N°_of_Shelf_i'].sum())
        Area_category = dataframe_clasification[dataframe_clasification['Company_Brand']== combobox_1]['Classification'].max()
        
    
    # Reverse Location from here
    # try:
            
    #     dataframe_2 = pd.read_csv("GUI_product_Database.csv")
    #     dataframe_2['dif'] = dataframe_2['Layout_width']-dataframe_2['p_width_f'] 

    #     if dataframe_2['dif'].count() > 0:
    #         dataframe_2.loc[(dataframe_2['dif'] >= p_width), 'Return'] = 1
    #         dataframe_2.loc[(dataframe_2['dif'] < p_width), 'Return'] = 0
            
    #         layout_num_min = layout_num
    #         Area_min = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Company_Brand']== combobox_1][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor]['Area'].max()

    #         return_sure = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['p_floor']==p_floor]['Return'].max()

    #         dataframe_1 = pd.read_csv("GUI_AI_2.csv") 
    #         Floor = dataframe_1['Floor'].values[0]
    #         Floor = int(Floor)
    #         p_floor_w_min = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['p_floor']==Floor][dataframe_2['Area']==Area_min]['p_width_f'].max() 
    #         Layout_width_min = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['p_floor']==Floor][dataframe_2['Area']==Area_min]['Layout_width'].max() 
            
    #         if(return_sure >-1):
    #             return_sure = return_sure
    #         else:
    #             return_sure = 0

    #         if(int(return_sure) != 0)&((p_floor_w_min + p_width)<Layout_width_min):

    #             layout_num_min = layout_num
    #             Area_min = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor]['Area'].max() 
    #             Area_width_min = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['Area_width'].min() 
    #             #Make p_floor transformation according to rules decision then calculate Area_min again

    #             if(Area_min>0):
    #                 if ((p_floor_w_min + p_width) > Layout_width_min) and (Floor == int(2)):
    #                     p_floor = Floor + 1
    #                 else:
    #                     p_floor = Floor

    #                 p_floor_w_min = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['p_width_f'].max() 
    #                 if p_floor == 1:
    #                     p_floor_1_w = p_floor_w_min + p_width
    #                     p_floor_2_w = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Area']==Area_min]['p_floor_2_w'].max() 
    #                     p_floor_3_w = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Area']==Area_min]['p_floor_3_w'].max() 
    #                     p_floor_h = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Area']==Area_min]['floor_1_h_i'].max()
    #                 elif p_floor == 2:
    #                     p_floor_2_w = p_floor_w_min + p_width
    #                     p_floor_1_w = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Area']==Area_min]['p_floor_1_w'].max() 
    #                     p_floor_3_w = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Area']==Area_min]['p_floor_3_w'].max() 
    #                     p_floor_h = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Area']==Area_min]['Floor_2_h_i'].max()
    #                 elif p_floor == 3:
    #                     p_floor_3_w = p_floor_w_min + p_width
    #                     p_floor_1_w = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Area']==Area_min]['p_floor_1_w'].max() 
    #                     p_floor_2_w = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Area']==Area_min]['p_floor_2_w'].max() 
    #                     p_floor_h = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Area']==Area_min]['Floor_3_h_i'].max()
                        
    #                 Area = Area_min
    #                 layout_num = layout_num_min 
    #                 Layout_width = Layout_width_min
    #                 Area_width = Area_width_min 
    #                 Layout_height = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['Layout_height'].max() 
    #                 floor_width = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['floor_width'].max() 
    #                 floor_height = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['floor_height'].max() 
    #                 floor_1_w_i = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['floor_1_w_i'].max() 
    #                 floor_1_h_i = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['floor_1_h_i'].max() 
    #                 floor_1_w_f = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['floor_1_w_f'].max() 
    #                 floor_1_h_f = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['floor_1_h_f'].max() 
    #                 Floor_2_w_i = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['Floor_2_w_i'].max() 
    #                 Floor_2_h_i = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['Floor_2_h_i'].max() 
    #                 Floor_2_w_f = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['Floor_2_w_f'].max() 
    #                 Floor_2_h_f = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['Floor_2_h_f'].max() 
    #                 Floor_3_w_i = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['Floor_3_w_i'].max() 
    #                 Floor_3_h_i = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['Floor_3_h_i'].max() 
    #                 Floor_3_w_f = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['Floor_3_w_f'].max() 
    #                 Floor_3_h_f = dataframe_2[dataframe_2['layout_num']==layout_num_min][dataframe_2['Return']== 1][dataframe_2['p_floor']==p_floor][dataframe_2['Area']==Area_min]['Floor_3_h_f'].max() 
    #                 p_width_i = p_floor_w_min
    #                 p_height_i = p_floor_h - p_height
    #                 p_width_f = p_width_i + p_width
    #                 p_height_f = p_height_i + p_height

    #                 Number = 1
    #                 try:
    #                     dataframe = pd.read_csv("GUI_product_Database.csv")
    #                     Number = dataframe[dataframe['layout_num']==layout_num]['Number'].max()
    #                     Number = 1 + Number
    #                     if(Number>0):
    #                         Number = Number
    #                     else:
    #                         Number = 1
    #                 except IndexError:
    #                     pass

    #                 p_code = str(Area)+"_"+str(p_floor)+"_"+str(Number)

    #             else:
    #                 pass

                
    #             af = pd.DataFrame({'layout_num':[layout_num],'Area':[Area],'Area_width':[Area_width],'Layout_width':[Layout_width],'Layout_height':[Layout_height],'floor_width':[floor_width],'floor_height':[floor_height],'floor_1_w_i':[floor_1_w_i],'floor_1_h_i':[floor_1_h_i],'floor_1_w_f':[floor_1_w_f],'floor_1_h_f':[floor_1_h_f],'Floor_2_w_i':[Floor_2_w_i],'Floor_2_h_i':[Floor_2_h_i],'Floor_2_w_f':[Floor_2_w_f],'Floor_2_h_f':[Floor_2_h_f],'Floor_3_w_i':[Floor_3_w_i],'Floor_3_h_i':[Floor_3_h_i],'Floor_3_w_f':[Floor_3_w_f],'Floor_3_h_f':[Floor_3_h_f],'p_width':[p_width],'p_height':[p_height],'p_weight':[p_weight],'p_floor':[p_floor],'p_floor_1_w':[p_floor_1_w],'p_floor_2_w':[p_floor_2_w],'p_floor_3_w':[p_floor_3_w],'p_width_i':[p_width_i],'p_height_i':[p_height_i],'p_width_f':[p_width_f],'p_height_f':[p_height_f],'Number':[Number],'p_code':[p_code],'Area_category':[Area_category],'Company_Brand':[combobox_1]})
    #             df = pd.read_csv("GUI_product_Database.csv")  
    #             df2 = pd.concat([af,df], ignore_index=True) 
    #             df2.to_csv('GUI_product_Database.csv',index = False,header=True)

    #             Filled = 0
    #             dataframe = pd.DataFrame({'layout_num': [layout_num],'Area': [Area],'p_floor': [p_floor],'Filled': [Filled]})
    #             dataframe.to_csv('Artifice_4.csv', mode='a', index=False, header=False)

    #             start_point_rect = (p_width_i+2, p_height_i)
    #             end_point_rect = (p_width_f, p_height_f)

    #             combobox_3 = "Priority" if a.get() else "Not Priority"
    #             if combobox_3 == "Priority":
    #                 red_color = (0, 0, 255) # Red color in BGR format
    #                 color = red_color
    #             elif combobox_3 == "Not Priority":
    #                 yellow_color = (0, 125, 255)
    #                 color = yellow_color

    #             thickness_rect = 2
    #             layout = cv2.imread(f'Layer{layout_num}_Area_{Area}.png')

    #             combobox_6 = "Fragile" if a.get() else "Not Fragile"
    #             if combobox_6 == "Fragile":
    #                 # Add Fragile Icon to the Product
    #                 fragile_icon = cv2.imread('fragile_2.jpg')
    #                 img1 = cv2.resize(fragile_icon,(30,25))
    #                 # x_end = width-int(width/10) + img1.shape[0]
    #                 y_end = p_height_i
    #                 x_end = int((p_width_i+p_width_f-2+img1.shape[1])/2)
    #                 # y_end = long-int(long/10) + img1.shape[1]
    #                 # blank_image[width-int(width/10):x_end,long-int(long/10):y_end]=img1 
    #                 layout[p_height_i-img1.shape[0]:y_end,x_end-img1.shape[1]:x_end] = img1
    #                 # img1 = layout[long-80:y_end,0:x_end]
    #             elif combobox_6 == "Not Fragile":
    #                 pass

    #             img_s= cv2.rectangle(layout, start_point_rect, end_point_rect, color, thickness_rect)
    #             image_saved = cv2.imwrite(f'Layer{layout_num}_Area_{Area}.png', img_s)
    #             # Add text to the Area
    #             text = f"{p_code}"
    #             font = cv2.FONT_HERSHEY_SIMPLEX
    #             font_scale = 0.6
    #             font_color = (0, 0, 0)  # Black color in BGR format
    #             font_thickness = 2
    #             text_position = (int((p_width_i+width/2-28)), int((p_height_i+long/2+5)))
    #             layout = cv2.imread(f'Layer{layout_num}_Area_{Area}.png')
    #             img_s= cv2.putText(
    #                 layout, text, text_position, font, font_scale, font_color, font_thickness
    #             )
    #             image_saved = cv2.imwrite(f'Layer{layout_num}_Area_{Area}.png', img_s)
    #             cv2.imshow("Image with Drawing", layout)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()

    #             return

    #         else:
    #             pass

    # except IndexError:
    #     pass
    # Reverse Location ends here



    dataframe_2 = pd.read_csv("GUI_product_Database.csv")    
    
    try:

        Area_1 = dataframe_2[dataframe_2['Area_category']==Area_category]['Area'][dataframe_2['p_floor']==Floor].max()  
        p_width_f = dataframe_2[dataframe_2['Area']==Area_1][dataframe_2['Area_category']==Area_category][dataframe_2['p_floor']==Floor]['p_width_f'].max()   
        Layout_width_1 = dataframe_2[dataframe_2['Area']==Area_1][dataframe_2['Area_category']==Area_category]['Layout_width'].max()

        if (Area_1>Area):
            Area = Area_1
        else:
            pass

        if ((p_width_f + p_width) > Layout_width_1):
            Area = Area_1 + 1
        else:
            pass

    except IndexError:
        pass


    
    dataframe = pd.read_csv("GUI_part4_distance.csv") 
    Area_max = dataframe['ID_Total'].max()
    layout_num = dataframe[dataframe['ID_Total']==Area]['layout_num'].max()

    if (Area > Area_max):
        blank_image = np.zeros((500, 700, 3), dtype=np.uint8)
        blank_image[:] = (64, 64, 64)  # White color in BGR format
        # Layout limit
        start_point_line = (0, 0)
        end_point_line = (700, 500)
        blue_color = (255, 0, 0)  # Blue color in BGR format
        thickness_line = 5
        img_s = cv2.rectangle(blank_image, start_point_line, end_point_line, blue_color, thickness_line)
        image_saved = cv2.imwrite('greetings.png', img_s)
        text = f"LIMIT REACHED FOR LEVEL {Floor}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 0, 0)  # Black color in BGR format
        font_thickness = 2
        text_position = (int((700/2-150)), int((500/2+5)))
        layout = cv2.imread('greetings.png')
        img_s= cv2.putText(
            layout, text, text_position, font, font_scale, font_color, font_thickness
        )
        image_saved = cv2.imwrite(f'greetings.png', img_s)
        text = f"Type {Area_category} Product"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 0, 0)  # Black color in BGR format
        font_thickness = 2
        text_position = (int((700/2-95)), int((500/2+35)))
        layout = cv2.imread('greetings.png')
        img_s= cv2.putText(
            layout, text, text_position, font, font_scale, font_color, font_thickness
        )
        image_saved = cv2.imwrite(f'greetings.png', img_s)

        afx = pd.DataFrame({'thislist':[f'greetings.png']})
        dfx = pd.read_csv("Images_save.csv")  
        df2x = pd.concat([afx,dfx], ignore_index=True) 
        df2x.to_csv('Images_save.csv',index = False,header=True)

        cv2.imshow("Image with Drawing", layout)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return print("LIMIT REACHED") 
    else:
        dataframe = pd.read_csv("GUI_part4_distance.csv") 
        width_x = dataframe[dataframe['ID_Total']==Area]['width_p'].max()
        long_x = dataframe[dataframe['ID_Total']==Area]['long_p'].max()

        if(width_x >= long_x):
            Area_width = width_x 
        else:
            Area_width = long_x

        Layout_width = int(Area_width*4.5)

    try:
        dataframe_x = pd.read_csv("GUI_product_Database.csv") 
        Area_x = dataframe_x['Area'].values[0]  
        if(Area_x>0):
            Area_x = Area_x
            
    except IndexError:
        Area_x = 0   
    
    dataframe = pd.read_csv("GUI_area_Database.csv")
    dataframe_2 = pd.read_csv("GUI_product_Database.csv")
 
    try:
        Area_y = dataframe[dataframe['Area']==Area]['Area'].max()
            
    except IndexError:
        pass
    
    if Area_x == 0:

        """Generate a shelf image with K levels"""
        Layout_height = 800
        floor_height = 15
        
        # Create blank white image
        blank_image = np.zeros((Layout_height, Layout_width, 3), dtype=np.uint8)
        blank_image[:] = (255, 255, 255)  # White background
        
        # Draw outer frame
        cv2.rectangle(blank_image, (0, 0), (Layout_width, Layout_height), (255, 0, 0), 5)
        
        # Calculate level positions dynamically
        for level in range(1, K_levels + 1):
            # Vertical position calculation
            level_height = int(Layout_height - ((level-1) * (Layout_height / K_levels)))
            
            # Draw shelf level
            cv2.rectangle(blank_image,
                        (0, level_height),  # Start point (left)
                        (Layout_width, level_height + floor_height),  # End point (right)
                        (255, 0, 0), -1)  # Blue fill
            
            # Add level label
            cv2.putText(blank_image, f"Level {level}", 
                    (10, level_height + 12),  # Position text just below shelf
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add title
        cv2.putText(blank_image, f'Layer {layout_num} Area {Area} - {K_levels} Levels',
                (Layout_width - 250, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save image
        cv2.imwrite(f'Layer{layout_num}_Area_{Area}_K{K_levels}.png', blank_image)
        
        afx = pd.DataFrame({'thislist':[f'Layer{layout_num}_Area_{Area}_K{K_levels}.png']})
        dfx = pd.read_csv("Images_save.csv")  
        df2x = pd.concat([afx,dfx], ignore_index=True) 
        df2x.to_csv('Images_save.csv',index = False,header=True)


        # Base metadata (same for all levels)
        config = {
            'layout_num': [layout_num],
            'Area': [Area],
            'Area_width': [Area_width],
            'Layout_width': [Layout_width],
            'Layout_height': [Layout_height],
            'floor_height': [floor_height],
            'K_levels': [K_levels],
            'Area_category':[Area_category]
        }
        
        # Dynamic level coordinates
        for level in range(1, K_levels + 1):
            level_height = int(Layout_height - ((level-1) * (Layout_height / K_levels)))
            
            config.update({
                f'Floor_{level}_w_i': [0],  # Always starts at left edge
                f'Floor_{level}_h_i': [level_height],
                f'Floor_{level}_w_f': [Layout_width],  # Always ends at right edge
                f'Floor_{level}_h_f': [level_height + floor_height]
            })

        af = pd.DataFrame(config)
        af.to_csv('GUI_area_Database.csv', index=False, header=True, mode='w')  # 'w' = write (explicit overwrite)   



        # Base metadata (same for all levels)
        config_2 = {
            'layout_num': [],
            'Area': [],
            'Area_width': [],
            'Layout_width': [],
            'Layout_height': [],
            'floor_height': [],
            'K_levels': [],
            'p_width':[],
            'p_height':[],
            'p_weight':[],
            'p_floor':[],
            'p_width_i':[],
            'p_height_i':[],
            'p_width_f':[],
            'p_height_f':[],
            'Number':[],
            'p_code':[],
            'Area_category':[],
            'Company_Brand':[]
        }
        
        # Dynamic level coordinates
        for level in range(1, K_levels + 1):
            level_height = int(Layout_height - ((level-1) * (Layout_height / K_levels)))
            
            config_2.update({
                f'Floor_{level}_w_i': [],  # Always starts at left edge
                f'Floor_{level}_h_i': [],
                f'Floor_{level}_w_f': [],  # Always ends at right edge
                f'Floor_{level}_h_f': [],
                f'p_floor_{level}_w': []
            })


        af_2 = pd.DataFrame(config_2)
        af_2.to_csv('GUI_product_Database.csv', index=False, header=True, mode='w')  # 'w' = write (explicit overwrite) 

    
    elif (Area_y != Area):
        
        """Generate a shelf image with K levels"""
        Layout_height = 800
        floor_height = 15
        
        # Create blank white image
        blank_image = np.zeros((Layout_height, Layout_width, 3), dtype=np.uint8)
        blank_image[:] = (255, 255, 255)  # White background
        
        # Draw outer frame
        cv2.rectangle(blank_image, (0, 0), (Layout_width, Layout_height), (255, 0, 0), 5)
        
        # Calculate level positions dynamically
        for level in range(1, K_levels + 1):
            # Vertical position calculation
            level_height = int(Layout_height - ((level-1) * (Layout_height / K_levels)))
            
            # Draw shelf level
            cv2.rectangle(blank_image,
                        (0, level_height),  # Start point (left)
                        (Layout_width, level_height + floor_height),  # End point (right)
                        (255, 0, 0), -1)  # Blue fill
            
            # Add level label
            cv2.putText(blank_image, f"Level {level}", 
                    (10, level_height + 12),  # Position text just below shelf
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add title
        cv2.putText(blank_image, f'Layer {layout_num} Area {Area} - {K_levels} Levels',
                (Layout_width - 250, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save image
        cv2.imwrite(f'Layer{layout_num}_Area_{Area}_K{K_levels}.png', blank_image)
        
        afx = pd.DataFrame({'thislist':[f'Layer{layout_num}_Area_{Area}_K{K_levels}.png']})
        dfx = pd.read_csv("Images_save.csv")  
        df2x = pd.concat([afx,dfx], ignore_index=True) 
        df2x.to_csv('Images_save.csv',index = False,header=True)


        # Base metadata (same for all levels)
        config = {
            'layout_num': [layout_num],
            'Area': [Area],
            'Area_width': [Area_width],
            'Layout_width': [Layout_width],
            'Layout_height': [Layout_height],
            'floor_height': [floor_height],
            'K_levels': [K_levels],
            'Area_category':[Area_category]
        }
        
        # Dynamic level coordinates
        for level in range(1, K_levels + 1):
            level_height = int(Layout_height - ((level-1) * (Layout_height / K_levels)))
            
            config.update({
                f'Floor_{level}_w_i': [0],  # Always starts at left edge
                f'Floor_{level}_h_i': [level_height],
                f'Floor_{level}_w_f': [Layout_width],  # Always ends at right edge
                f'Floor_{level}_h_f': [level_height + floor_height]
            })

        af = pd.DataFrame(config)
        df = pd.read_csv("GUI_area_Database.csv")  
        df2 = pd.concat([af,df], ignore_index=True) 
        df2.to_csv('GUI_area_Database.csv',index = False,header=True)
        
        
    else:
        
        pass
    
    Company_Brand = combobox_1

    dataframe_2 = pd.read_csv("GUI_area_Database.csv") 
    Area = dataframe_2[dataframe_2['Area']==Area]['Area'].max()    
    Area_width = dataframe_2[dataframe_2['Area']==Area]['Area_width'].max()
    Layout_width = dataframe_2[dataframe_2['Area']==Area]['Layout_width'].max()
    Layout_height = dataframe_2[dataframe_2['Area']==Area]['Layout_height'].max()
    floor_height = dataframe_2[dataframe_2['Area']==Area]['floor_height'].max()
    

    # Initialize dictionaries to store the values
    Floor_w_i_ = {}
    Floor_h_i_ = {}
    Floor_w_f_ = {}
    Floor_h_f_ = {}
    p_floor_w_ = {}    

    for level in range(1, K_levels + 1):  # +1 if K_levels is inclusive
        # Get max values for each parameter
        Floor_w_i_[level] = dataframe_2[dataframe_2['Area'] == Area][f'Floor_{level}_w_i'].max()
        Floor_h_i_[level] = dataframe_2[dataframe_2['Area'] == Area][f'Floor_{level}_h_i'].max()
        Floor_w_f_[level] = dataframe_2[dataframe_2['Area'] == Area][f'Floor_{level}_w_f'].max()
        Floor_h_f_[level] = dataframe_2[dataframe_2['Area'] == Area][f'Floor_{level}_h_f'].max()
        
   
    print(Area)
    print(Area_width)

    width = int(Width.get())
    long = int(Long.get())
    weight = int(Weight_1.get())

    p_width = width    
    p_height = long
    p_weight = weight 

    p_floor = Floor 

    try:
        dataframe = pd.read_csv("GUI_product_Database.csv") 

        for level in range(1, K_levels + 1):  # +1 if K_levels is inclusive
            # Get max values for each parameter
            p_floor_w_[level] = dataframe[dataframe['Area'] == Area][f'p_floor_{level}_w'].max()

    except IndexError:
        for level in range(1, K_levels + 1):  # +1 if K_levels is inclusive
            # Get max values for each parameter
            p_floor_w_[level] = 0

    for level in range(1, K_levels + 1):  # +1 if K_levels is inclusive
        # Get max values for each parameter
        if p_floor_w_[level] >= 0:
            pass
        else:
            p_floor_w_[level] = 0

    for level in range(1, K_levels + 1):  # +1 if K_levels is inclusive
        # Get max values for each parameter
        if p_floor == level:
            p_floor_w_[level] = p_width + p_floor_w_[level]
        else:   
            p_floor_w_[level] = 0 + p_floor_w_[level]

        if p_floor == level:
            p_floor_w = p_floor_w_[level] - p_width
            p_floor_h = Floor_h_i_[level]
        else:
            pass
        
        if (p_floor == level):
            if (p_floor_w + p_width) > Layout_width:
                Area = Area + 1
                # p_floor = p_floor
                p_floor_w_[level] = p_width
                p_floor_w = p_floor_w_[level] - p_width
                p_floor_h = Floor_h_i_[level]    

                try:
                    dataframe_2 = pd.read_csv("GUI_product_Database.csv")
                    Area_y = dataframe_2[dataframe_2['Area']==Area]['Area'].max() 
                    for level_2 in range(1, K_levels + 1):  # +1 if K_levels is inclusive
                        # Get max values for each parameter
                        if p_floor != level_2:
                            p_floor_w_[level_2] = dataframe_2[dataframe_2['Area']==Area][f'p_floor_{level_2}_w'].max()  
                except IndexError:
                        pass
                for level_2 in range(1, K_levels + 1):  # +1 if K_levels is inclusive
                    # Get max values for each parameter
                    if (Area_y != Area)&(p_floor != level_2):
                        p_floor_w_[level_2] = 0


                try: 
                    dataframe_2 = pd.read_csv("GUI_product_Database.csv") 
                    Area_y = dataframe_2[dataframe_2['Area']==Area]['Area'].max()

                    if (Area_y==Area):    
                        # p_floor = level
                        dataframe_2 = pd.read_csv("GUI_product_Database.csv")
                        p_floor_w_[level] = dataframe_2[dataframe_2['Area']==Area][f'p_floor_{level}_w'].max()
                        p_floor_w_[level] = p_width + p_floor_w_[level]
                        p_floor_w = p_floor_w_[level] - p_width
                        p_floor_h = Floor_h_i_[level]    

                        for level_2 in range(1, K_levels + 1):  # +1 if K_levels is inclusive
                            # Get max values for each parameter
                            if p_floor != level_2:
                                p_floor_w_[level_2] = dataframe_2[dataframe_2['Area']==Area][f'p_floor_{level_2}_w'].max()  

                        Layout_width = dataframe_2[dataframe_2['Area']==Area]['Layout_width'].max()
                        
                        if (p_floor_w_[level] > Layout_width):

                            dataframe = pd.read_csv("GUI_part4_distance.csv") 
                            Area_max = dataframe['ID_Total'].max()

                            while (Area<=Area_max) or (p_floor_w_[level] > Layout_width):
                                
                                Area = Area +1
                                # p_floor = level
                                p_floor_w_[level] = dataframe_2[dataframe_2['Area']==Area][f'p_floor_{level}_w'].max()
                                p_floor_w_[level] = p_width + p_floor_w_[level]
                                p_floor_w = p_floor_w_[level] - p_width
                                p_floor_h = Floor_h_i_[level]
                                

                                for level_2 in range(1, K_levels + 1):  # +1 if K_levels is inclusive
                                    # Get max values for each parameter
                                    if p_floor != level_2:
                                        p_floor_w_[level_2] = dataframe_2[dataframe_2['Area']==Area][f'p_floor_{level_2}_w'].max() 

                                Layout_width = dataframe_2[dataframe_2['Area']==Area]['Layout_width'].max()
                                    
                                Area_y = dataframe_2[dataframe_2['Area']==Area]['Area'].max()

                                if(Area_y!=Area):

                                    p_floor_w_[level] = 0
                                    p_floor_w_[level] = p_width + p_floor_w_[level] 
                                    p_floor_w = p_floor_w_[level] - p_width
                                    p_floor_h = Floor_h_i_[level]

                                    for level_2 in range(1, K_levels + 1):  # +1 if K_levels is inclusive
                                        # Get max values for each parameter
                                        if p_floor != level_2:
                                            p_floor_w_[level_2] = 0

                                    # break
                                else:
                                    pass

                            print(Area)
                        else:
                            pass

                    else:
                        pass
                except IndexError:
                    pass
        
        dataframe = pd.read_csv("GUI_part4_distance.csv") 
        Area_max = dataframe['ID_Total'].max()

        if (Area > Area_max):
            blank_image = np.zeros((500, 700, 3), dtype=np.uint8)
            blank_image[:] = (64, 64, 64)  # White color in BGR format
            # Layout limit
            start_point_line = (0, 0)
            end_point_line = (700, 500)
            blue_color = (255, 0, 0)  # Blue color in BGR format
            thickness_line = 5
            img_s = cv2.rectangle(blank_image, start_point_line, end_point_line, blue_color, thickness_line)
            image_saved = cv2.imwrite('greetings.png', img_s)
            text = f"LIMIT REACHED FOR LEVEL {Floor}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (0, 0, 0)  # Black color in BGR format
            font_thickness = 2
            text_position = (int((700/2-150)), int((500/2+5)))
            layout = cv2.imread('greetings.png')
            img_s= cv2.putText(
                layout, text, text_position, font, font_scale, font_color, font_thickness
            )
            image_saved = cv2.imwrite(f'greetings.png', img_s)
            text = f"Type {Area_category} Product"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (0, 0, 0)  # Black color in BGR format
            font_thickness = 2
            text_position = (int((700/2-95)), int((500/2+35)))
            layout = cv2.imread('greetings.png')
            img_s= cv2.putText(
                layout, text, text_position, font, font_scale, font_color, font_thickness
            )
            image_saved = cv2.imwrite(f'greetings.png', img_s)

            afx = pd.DataFrame({'thislist':[f'greetings.png']})
            dfx = pd.read_csv("Images_save.csv")  
            df2x = pd.concat([afx,dfx], ignore_index=True) 
            df2x.to_csv('Images_save.csv',index = False,header=True)

            cv2.imshow("Image with Drawing", layout)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return print("LIMIT REACHED") 
        else:
            dataframe = pd.read_csv("GUI_part4_distance.csv") 
            width_x = dataframe[dataframe['ID_Total']==Area]['width_p'].max()
            long_x = dataframe[dataframe['ID_Total']==Area]['long_p'].max()

            if(width_x >= long_x):
                Area_width = width_x 
            else:
                Area_width = long_x

            Layout_width = int(Area_width*4.5)


        dataframe_2 = pd.read_csv("GUI_area_Database.csv") 
        try:
            Area_y = dataframe_2[dataframe_2['Area']==(Area)]['Area'].max()
        except IndexError:
            pass
                    

        if (Area_y != Area):
            
            """Generate a shelf image with K levels"""
            Layout_height = 800
            floor_height = 15
            
            # Create blank white image
            blank_image = np.zeros((Layout_height, Layout_width, 3), dtype=np.uint8)
            blank_image[:] = (255, 255, 255)  # White background
            
            # Draw outer frame
            cv2.rectangle(blank_image, (0, 0), (Layout_width, Layout_height), (255, 0, 0), 5)
            
            # Calculate level positions dynamically
            for level in range(1, K_levels + 1):
                # Vertical position calculation
                level_height = int(Layout_height - ((level-1) * (Layout_height / K_levels)))
                
                # Draw shelf level
                cv2.rectangle(blank_image,
                            (0, level_height),  # Start point (left)
                            (Layout_width, level_height + floor_height),  # End point (right)
                            (255, 0, 0), -1)  # Blue fill
                
                # Add level label
                cv2.putText(blank_image, f"Level {level}", 
                        (10, level_height + 12),  # Position text just below shelf
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Add title
            cv2.putText(blank_image, f'Layer {layout_num} Area {Area} - {K_levels} Levels',
                    (Layout_width - 250, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Save image
            cv2.imwrite(f'Layer{layout_num}_Area_{Area}_K{K_levels}.png', blank_image)
            
            afx = pd.DataFrame({'thislist':[f'Layer{layout_num}_Area_{Area}_K{K_levels}.png']})
            dfx = pd.read_csv("Images_save.csv")  
            df2x = pd.concat([afx,dfx], ignore_index=True) 
            df2x.to_csv('Images_save.csv',index = False,header=True)


            # Base metadata (same for all levels)
            config = {
                'layout_num': [layout_num],
                'Area': [Area],
                'Area_width': [Area_width],
                'Layout_width': [Layout_width],
                'Layout_height': [Layout_height],
                'floor_height': [floor_height],
                'K_levels': [K_levels],
                'Area_category':[Area_category]
            }
            
            # Dynamic level coordinates
            for level in range(1, K_levels + 1):
                level_height = int(Layout_height - ((level-1) * (Layout_height / K_levels)))
                
                config.update({
                    f'Floor_{level}_w_i': [0],  # Always starts at left edge
                    f'Floor_{level}_h_i': [level_height],
                    f'Floor_{level}_w_f': [Layout_width],  # Always ends at right edge
                    f'Floor_{level}_h_f': [level_height + floor_height]
                })

            af = pd.DataFrame(config)
            df = pd.read_csv("GUI_area_Database.csv")  
            df2 = pd.concat([af,df], ignore_index=True) 
            df2.to_csv('GUI_area_Database.csv',index = False,header=True)

        else:
            pass


    p_width_i = p_floor_w
    p_height_i = p_floor_h - p_height
    p_width_f = p_width_i + p_width
    p_height_f = p_height_i + p_height

    Number = 1
    try:
        dataframe = pd.read_csv("GUI_product_Database.csv")
        Number = dataframe['Number'].max()
        Number = 1 + Number
        if(Number>0):
            Number = Number
        else:
            Number = 1
    except IndexError:
        pass

    p_code = str(Area)+"_"+str(p_floor)+"_"+str(Number)
    



    # Base metadata (same for all levels)
    config_2 = {
        'layout_num': [layout_num],
        'Area': [Area],
        'Area_width': [Area_width],
        'Layout_width': [Layout_width],
        'Layout_height': [Layout_height],
        'floor_height': [floor_height],
        'K_levels': [K_levels],
        'p_width':[p_width],
        'p_height':[p_height],
        'p_weight':[p_weight],
        'p_floor':[p_floor],
        'p_width_i':[p_width_i],
        'p_height_i':[p_height_i],
        'p_width_f':[p_width_f],
        'p_height_f':[p_height_f],
        'Number':[Number],
        'p_code':[p_code],
        'Area_category':[Area_category],
        'Company_Brand':[Company_Brand]
    }
    
    # Dynamic level coordinates
    for level in range(1, K_levels + 1):
        level_height = int(Layout_height - ((level-1) * (Layout_height / K_levels)))
        
        config_2.update({
            f'Floor_{level}_w_i': [Floor_w_i_[level]],  # Always starts at left edge
            f'Floor_{level}_h_i': [Floor_h_i_[level]],
            f'Floor_{level}_w_f': [Floor_w_f_[level]],  # Always ends at right edge
            f'Floor_{level}_h_f': [Floor_h_f_[level]],
            f'p_floor_{level}_w': [p_floor_w_[level]]
        })

    af = pd.DataFrame(config_2)
    df = pd.read_csv("GUI_product_Database.csv")  
    df2 = pd.concat([af,df], ignore_index=True) 
    df2.to_csv('GUI_product_Database.csv',index = False,header=True)

    

    start_point_rect = (p_width_i+2, p_height_i)
    end_point_rect = (p_width_f, p_height_f)

    combobox_3 = "Priority" if b.get() else "Not Priority"
    if combobox_3 == "Priority":
        red_color = (0, 0, 255) # Red color in BGR format
        color = red_color
    elif combobox_3 == "Not Priority":
        yellow_color = (0, 125, 255)
        color = yellow_color

    thickness_rect = 2
    layout = cv2.imread(f'Layer{layout_num}_Area_{Area}_K{K_levels}.png')

    combobox_6 = "Fragile" if a.get() else "Not Fragile"
    if combobox_6 == "Fragile":
        # Add Fragile Icon to the Product
        fragile_icon = cv2.imread('fragile_2.jpg')
        img1 = cv2.resize(fragile_icon,(30,25))
        # x_end = width-int(width/10) + img1.shape[0]
        y_end = p_height_i
        x_end = int((p_width_i+p_width_f-2+img1.shape[1])/2)
        # y_end = long-int(long/10) + img1.shape[1]
        # blank_image[width-int(width/10):x_end,long-int(long/10):y_end]=img1 
        layout[p_height_i-img1.shape[0]:y_end,x_end-img1.shape[1]:x_end] = img1
        # img1 = layout[long-80:y_end,0:x_end]
    elif combobox_6 == "Not Fragile":
        pass

    img_s= cv2.rectangle(layout, start_point_rect, end_point_rect, color, thickness_rect)
    image_saved = cv2.imwrite(f'Layer{layout_num}_Area_{Area}.png', img_s)
    # Add text to the Area
    text = f"{p_code}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 0, 0)  # Black color in BGR format
    font_thickness = 2
    text_position = (int((p_width_i+width/2-28)), int((p_height_i+long/2+5)))
    layout = cv2.imread(f'Layer{layout_num}_Area_{Area}.png')
    img_s= cv2.putText(
        layout, text, text_position, font, font_scale, font_color, font_thickness
    )
    image_saved = cv2.imwrite(f'Layer{layout_num}_Area_{Area}_K{K_levels}.png', img_s)

    afx = pd.DataFrame({'thislist':[f'Layer{layout_num}_Area_{Area}_K{K_levels}.png']})
    dfx = pd.read_csv("Images_save.csv")  
    df2x = pd.concat([afx,dfx], ignore_index=True) 
    df2x.to_csv('Images_save.csv',index = False,header=True)

    cv2.imshow("Image with Drawing", layout)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

     
    
button = customtkinter.CTkButton(widgets_frame, text="Insert Product", command=area_product_number)
button.grid(row=8, column=0, padx=5, pady=5, sticky="nsew")


widgets_frame = customtkinter.CTkLabel(frame, text="Product Categories")
widgets_frame.grid(row=9, column=0, padx=5, pady=5)

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def chart():

    #define aesthetics for plot
    color1 = 'steelblue'

    from datetime import datetime

    df = pd.read_csv("Static_Training_Dataset_2.csv",sep = ',')

    df['Time_start'] = pd.to_datetime(df['Time_start'])
    df['Time_finish'] = pd.to_datetime(df['Time_finish'])

    df['Time_spent'] = df['Time_finish'] - df['Time_start'] 

    df_2= df.groupby("Company_Brand")['Time_spent'].sum()

    df_2 = pd.DataFrame({'Company_Brand':df_2.index, 'Sum_Time':df_2.values})

    df_2 = df_2.sort_values(by='Sum_Time', ascending=True)
    Total_Sum_Time = df_2['Sum_Time'].sum()
    df_2['%Time'] = df_2['Sum_Time']/Total_Sum_Time
    df_2['%Acum_Time'] = df_2['%Time'].cumsum(axis=0)
    df_2.loc[df_2['%Acum_Time'] <= 0.1, 'Pareto_Time'] = 'A' 
    df_2.loc[(df_2['%Acum_Time'] <= 0.25) & (df_2['%Acum_Time'] > 0.1), 'Pareto_Time'] = 'B'
    df_2.loc[(df_2['%Acum_Time'] <= 1) & (df_2['%Acum_Time'] > 0.25), 'Pareto_Time'] = 'C' 

    df_2.loc[df_2['Pareto_Time'] == 'A', 'Pareto_Time_num'] = 3
    df_2.loc[df_2['Pareto_Time'] == 'B', 'Pareto_Time_num'] = 2
    df_2.loc[df_2['Pareto_Time'] == 'C', 'Pareto_Time_num'] = 1

    df_2.to_csv('Pareto_Time.csv',index = False,header=True)

    df_2_y = pd.read_csv("Pareto_Time.csv")
    s = {}
    df_2_x = pd.DataFrame(s)
    df_2_x['Time_num'] = range(1, 1 + len(df_2_y))
    df_2_x = df_2_x.sort_values(by='Time_num', ascending=False)
    df_2_x = df_2_x.reset_index()
    df_2_y['Time_num'] = df_2_x['Time_num'] 

    df_3= df.groupby("Company_Brand")['Company_Brand'].count()

    df_3 = pd.DataFrame({'Company_Brand':df_3.index, 'Count':df_3.values})

    df_3 = df_3.sort_values(by='Count', ascending=False)
    Total_Count = df_3['Count'].sum()
    df_3['%_Frequency'] = df_3['Count']/Total_Count
    df_3['%Acum_Frequency'] = df_3['%_Frequency'].cumsum(axis=0)
    df_3.loc[df_3['%Acum_Frequency'] <= 0.75, 'Pareto_Frequency'] = 'A' 
    df_3.loc[(df_3['%Acum_Frequency'] <= 0.90) & (df_3['%Acum_Frequency'] > 0.75), 'Pareto_Frequency'] = 'B'
    df_3.loc[(df_3['%Acum_Frequency'] <= 1) & (df_3['%Acum_Frequency'] > 0.90), 'Pareto_Frequency'] = 'C' 

    df_3.loc[df_3['Pareto_Frequency'] == 'A', 'Pareto_Frequency_num'] = 3
    df_3.loc[df_3['Pareto_Frequency'] == 'B', 'Pareto_Frequency_num'] = 2
    df_3.loc[df_3['Pareto_Frequency'] == 'C', 'Pareto_Frequency_num'] = 1

    df_3.to_csv('Pareto_Frequency.csv',index = False,header=True)

    df_3_y = pd.read_csv("Pareto_Frequency.csv")
    s = {}
    df_3_x = pd.DataFrame(s)
    df_3_x['Frequency_num'] = range(1, 1 + len(df_3_y))
    df_3_x = df_3_x.sort_values(by='Frequency_num', ascending=False)
    df_3_x = df_3_x.reset_index()
    df_3_y['Frequency_num'] = df_3_x['Frequency_num'] 


    df_4 = pd.merge(df_2, df_3, on=['Company_Brand'])
    df_4 = df_4[['Company_Brand','Sum_Time','Count']]
    df_4['Time_per_spent'] = df_4['Sum_Time']/df_4['Count']
    df_4 = df_4.sort_values(by='Time_per_spent', ascending=True)
    Total_Time_per_spent = df_4['Time_per_spent'].sum()
    df_4['%Time_per_spent'] = df_4['Time_per_spent']/Total_Time_per_spent
    df_4.drop(['Sum_Time', 'Count'], axis=1)
    df_4['%Acum_Time_per_spent'] = df_4['%Time_per_spent'].cumsum(axis=0)
    df_4.loc[df_4['%Acum_Time_per_spent'] <= 0.10, 'Pareto_Time_per_spent'] = 'A' 
    df_4.loc[(df_4['%Acum_Time_per_spent'] <= 0.25) & (df_4['%Acum_Time_per_spent'] > 0.10), 'Pareto_Time_per_spent'] = 'B'
    df_4.loc[(df_4['%Acum_Time_per_spent'] <= 1) & (df_4['%Acum_Time_per_spent'] > 0.25), 'Pareto_Time_per_spent'] = 'C' 

    df_4.loc[df_4['Pareto_Time_per_spent'] == 'A', 'Pareto_Time_per_spent_num'] = 3
    df_4.loc[df_4['Pareto_Time_per_spent'] == 'B', 'Pareto_Time_per_spent_num'] = 2
    df_4.loc[df_4['Pareto_Time_per_spent'] == 'C', 'Pareto_Time_per_spent_num'] = 1

    df_4.to_csv('Pareto_Time_per_spent.csv',index = False,header=True)


    df_4_y = pd.read_csv("Pareto_Time_per_spent.csv")
    s = {}
    df_4_x = pd.DataFrame(s)
    df_4_x['Time_per_spent_num'] = range(1, 1 + len(df_4_y))
    df_4_x = df_4_x.sort_values(by='Time_per_spent_num', ascending=False)
    df_4_x = df_4_x.reset_index()
    df_4_y['Time_per_spent_num'] = df_4_x['Time_per_spent_num'] 

    df_5 = df.groupby("Company_Brand")['Income'].sum()

    df_5 = pd.DataFrame({'Company_Brand':df_5.index, 'Sum_Prof':df_5.values})

    df_5 = df_5.sort_values(by='Sum_Prof', ascending=False)
    Total_Prof = df_5['Sum_Prof'].sum()
    df_5['%Profitable'] = df_5['Sum_Prof']/Total_Prof
    df_5['%Acum_Profitable'] = df_5['%Profitable'].cumsum(axis=0)
    df_5.loc[df_5['%Acum_Profitable'] <= 0.75, 'Pareto_Profitability'] = 'A' 
    df_5.loc[(df_5['%Acum_Profitable'] <= 0.90) & (df_5['%Acum_Profitable'] > 0.75), 'Pareto_Profitability'] = 'B'
    df_5.loc[(df_5['%Acum_Profitable'] <= 1) & (df_5['%Acum_Profitable'] > 0.90), 'Pareto_Profitability'] = 'C' 

    df_5.loc[df_5['Pareto_Profitability'] == 'A', 'Pareto_Profitability_num'] = 3
    df_5.loc[df_5['Pareto_Profitability'] == 'B', 'Pareto_Profitability_num'] = 2
    df_5.loc[df_5['Pareto_Profitability'] == 'C', 'Pareto_Profitability_num'] = 1

    df_5.to_csv('Pareto_Profitability.csv',index = False,header=True)

    df_5_y = pd.read_csv("Pareto_Profitability.csv")
    s = {}
    df_5_x = pd.DataFrame(s)
    df_5_x['Profitability_num'] = range(1, 1 + len(df_5_y))
    df_5_x = df_5_x.sort_values(by='Profitability_num', ascending=False)
    df_5_x = df_5_x.reset_index()
    df_5_y['Profitability_num'] = df_5_x['Profitability_num'] 


    data = pd.read_csv("weights.csv")
    weight_time = data[data['weight_type']=='time weight']['weight'].max() 
    weight_frequency = data[data['weight_type']=='frequency weight']['weight'].max() 
    weight_time_per_spent = data[data['weight_type']=='per item time spent weight']['weight'].max() 
    weight_profitability = data[data['weight_type']=='profit weight']['weight'].max()     


    df_6 = pd.merge(df_2, df_3, on=['Company_Brand'])
    df_6 = pd.merge(df_4, df_6, on=['Company_Brand'])
    df_6 = pd.merge(df_5, df_6, on=['Company_Brand'])

    s = {}
    df_6_x = pd.DataFrame(s)
    if(weight_time > 0):
        df_6_x['Pareto_Time'] = df_6['Pareto_Time']
    else:
        df_6_x['Pareto_Time'] = None
    if(weight_frequency > 0):
        df_6_x['Pareto_Frequency'] = df_6['Pareto_Frequency']
    else:
        df_6_x['Pareto_Frequency'] = None
    if(weight_time_per_spent > 0):
        df_6_x['Pareto_Time_per_spent'] = df_6['Pareto_Time_per_spent']
    else:
        df_6_x['Pareto_Time_per_spent'] = None
    if(weight_profitability > 0):
        df_6_x['Pareto_Profitability'] = df_6['Pareto_Profitability']
    else:
        df_6_x['Pareto_Profitability'] = None

    df_6_x.fillna('', inplace=True)

    df_6['Multi_Pareto_Classification'] = df_6_x['Pareto_Time'] + df_6_x['Pareto_Frequency'] + df_6_x['Pareto_Time_per_spent'] + df_6_x['Pareto_Profitability']

    df_6 = df_6[['Company_Brand', 'Pareto_Time_per_spent_num', 'Pareto_Profitability_num', 'Pareto_Frequency_num', 'Pareto_Time_num','Multi_Pareto_Classification']]
    df_6['Multi_Pareto_Result'] = df_6['Pareto_Time_num']*weight_time + df_6['Pareto_Frequency_num']*weight_frequency + df_6['Pareto_Time_per_spent_num']*weight_time_per_spent + df_6['Pareto_Profitability_num']*weight_profitability
    df_6 = df_6.sort_values(by='Multi_Pareto_Result', ascending=False)
    df_6.to_csv('Multi_Pareto.csv',index = False,header=True)

    df_7 = df_6[['Company_Brand', 'Multi_Pareto_Result']]
    df_7 = pd.merge(df_7, df_3, on=['Company_Brand'])
    df_7 = df_7[['Company_Brand', 'Multi_Pareto_Result', '%_Frequency']]
    dataframe = pd.read_csv("GUI_part4_distance.csv") 
    layout_num_maxi = dataframe['layout_num'].max()
    Shelfs_max = dataframe[dataframe['layout_num']==layout_num_maxi]['ID_Total'].max()
    df_7['Shelves'] = df_7['%_Frequency']*Shelfs_max
    df_7['Acum_Shelf'] = df_7['Shelves'].cumsum(axis=0)
    df_7.loc[(df_7['Acum_Shelf'].astype(int) == Shelfs_max), 'N°_of_Shelf_f_prev'] = (df_7['Acum_Shelf']).astype(int)
    df_7.loc[(df_7['Acum_Shelf'] < Shelfs_max), 'N°_of_Shelf_f_prev'] = (df_7['Acum_Shelf']).astype(int)+1

    df_8_x = df_7.groupby("Multi_Pareto_Result")['N°_of_Shelf_f_prev'].max()
    df_8_x = pd.DataFrame({'Multi_Pareto_Result':df_8_x.index, 'N°_of_Shelf_f':df_8_x.values})
    df_8_x = df_8_x.sort_values(by='Multi_Pareto_Result', ascending=False)
    df_8_x.to_csv('Multi_Pareto_Shelf_prev.csv',index = False,header=True)
    df_8 = pd.read_csv("Multi_Pareto_Shelf_prev.csv")

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice.csv",index = False,header=True)

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice_2.csv",index = False,header=True)

    first_row = df_8['N°_of_Shelf_f'].values[0]
    first_multi_pareto_result = df_8['Multi_Pareto_Result'].values[0]

    def get_shelf_i(shelf_i):
        y = -1

        dataframe = pd.read_csv("artifice.csv")
        artifice = dataframe['artifice'].values[0]

        while(y<artifice-1):
            y = y+1

        if (shelf_i == first_row):
            if(first_multi_pareto_result == df_8['Multi_Pareto_Result'].values[y]):
                dataframe = pd.read_csv("artifice.csv")
                dataframe = dataframe.drop(['artifice'], axis=1)
                dataframe.insert(0,"artifice",[artifice+1],False)
                dataframe.to_csv("artifice.csv",index = False,header=True)
                return 1
            else:
                s={'artifice':[1]}
                dataframe = pd.DataFrame(s)
                dataframe.to_csv("artifice.csv",index = False,header=True)
                return df_8['N°_of_Shelf_f'].values[y]

        else:
            i = -1

            dataframe = pd.read_csv("artifice_2.csv")
            artifice = dataframe['artifice'].values[0]
            while(i<artifice-1):
                i = i+1
                    
                dataframe = pd.read_csv("artifice_2.csv")
                dataframe = dataframe.drop(['artifice'], axis=1)
                dataframe.insert(0,"artifice",[artifice+1],False)
                dataframe.to_csv("artifice_2.csv",index = False,header=True)
            
            if (df_8['Multi_Pareto_Result'].values[i]==df_8['Multi_Pareto_Result'].values[i-1]):
                return df_8['N°_of_Shelf_f'].values[i-1]        
            else:        
                return df_8['N°_of_Shelf_f'].values[i]
          

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice.csv",index = False,header=True)

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice_2.csv",index = False,header=True)

    df_8['N°_of_Shelf_i'] = df_8['N°_of_Shelf_f'].apply(get_shelf_i)

    df_7 = pd.merge(df_7, df_8, on=['Multi_Pareto_Result'])
    df_7.to_csv('Multi_Pareto_Shelf.csv',index = False,header=True)

    df_6_y = pd.merge(df_2_y, df_3_y, on=['Company_Brand'])
    df_6_y = pd.merge(df_4_y, df_6_y, on=['Company_Brand'])
    df_6_y = pd.merge(df_5_y, df_6_y, on=['Company_Brand'])
    df_6_y = df_6_y[['Company_Brand', 'Time_per_spent_num', 'Profitability_num', 'Frequency_num', 'Time_num']]
    df_6_y['Distinct_Result'] = df_6_y['Time_num']*weight_time + df_6_y['Frequency_num']*weight_frequency + df_6_y['Time_per_spent_num']*weight_time_per_spent + df_6_y['Profitability_num']*weight_profitability
    df_6_y = df_6_y.sort_values(by='Distinct_Result', ascending=False)
    df_6_y.to_csv('Distinct_Result.csv',index = False,header=True)

    df_7_y = df_6_y[['Company_Brand', 'Distinct_Result']]
    df_7_y = pd.merge(df_7_y, df_3_y, on=['Company_Brand'])
    df_7_y = df_7_y[['Company_Brand', 'Distinct_Result', '%_Frequency']]
    dataframe = pd.read_csv("GUI_part4_distance.csv")
    layout_num = dataframe['layout_num'].max() 
    Shelfs_max = dataframe[dataframe['layout_num']==layout_num]['ID'].max()
    df_7_y['Shelves'] = df_7_y['%_Frequency']*Shelfs_max
    df_7_y['Acum_Shelf'] = df_7_y['Shelves'].cumsum(axis=0)
    df_7_y.loc[((df_7_y['Acum_Shelf']).astype(int) == Shelfs_max), 'N°_of_Shelf_f_prev'] = (df_7_y['Acum_Shelf']).astype(int)
    df_7_y.loc[(df_7_y['Acum_Shelf'] < Shelfs_max), 'N°_of_Shelf_f_prev'] = (df_7_y['Acum_Shelf']).astype(int)+1

    df_8_y_2 = df_7_y.groupby("Distinct_Result")['N°_of_Shelf_f_prev'].max()
    df_8_y_2 = pd.DataFrame({'Distinct_Result':df_8_y_2.index, 'N°_of_Shelf_f':df_8_y_2.values})
    df_8_y_2 = df_8_y_2.sort_values(by='Distinct_Result', ascending=False)
    df_8_y = df_8_y_2.reset_index()
    first_row = df_8_y['N°_of_Shelf_f'].values[0]
    first_distinct_result = df_8_y['Distinct_Result'].values[0]

    def get_shelf_i(shelf_i):
        y = -1

        dataframe = pd.read_csv("artifice.csv")
        artifice = dataframe['artifice'].values[0]

        while(y<artifice-1):
            y = y+1

        if (shelf_i == first_row) :
            if(first_distinct_result == df_8_y['Distinct_Result'].values[y]):
                dataframe = pd.read_csv("artifice.csv")
                dataframe = dataframe.drop(['artifice'], axis=1)
                dataframe.insert(0,"artifice",[artifice+1],False)
                dataframe.to_csv("artifice.csv",index = False,header=True)
                return 1
            else:
                s={'artifice':[1]}
                dataframe = pd.DataFrame(s)
                dataframe.to_csv("artifice.csv",index = False,header=True)
                return df_8_y['N°_of_Shelf_f'].values[y]

        else:
            i = -1

            dataframe = pd.read_csv("artifice_2.csv")
            artifice = dataframe['artifice'].values[0]
            
            while(i<artifice-1):
                i = i+1
                dataframe = pd.read_csv("artifice_2.csv")
                dataframe = dataframe.drop(['artifice'], axis=1)
                dataframe.insert(0,"artifice",[artifice+1],False)
                dataframe.to_csv("artifice_2.csv",index = False,header=True)
            
            data = pd.read_csv("weights.csv")
            weight_time = data[data['weight_type']=='time weight']['weight'].max() 
            weight_frequency = data[data['weight_type']=='frequency weight']['weight'].max()
            
            if(weight_time > 0)|(weight_frequency > 0):
                if (df_8_y['Distinct_Result'].values[i]==df_8_y['Distinct_Result'].values[i+1]):
                    return df_8_y['N°_of_Shelf_f'].values[i]        
                else: 
                    return df_8_y['N°_of_Shelf_f'].values[i+1]
            else:
                if (df_8_y['Distinct_Result'].values[i]==df_8_y['Distinct_Result'].values[i-1]):
                    return df_8_y['N°_of_Shelf_f'].values[i-1]        
                else: 
                    return df_8_y['N°_of_Shelf_f'].values[i]
                

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice.csv",index = False,header=True)

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice_2.csv",index = False,header=True)

    df_8_y['N°_of_Shelf_i'] = df_8_y['N°_of_Shelf_f'].apply(get_shelf_i)

    df_7_y = pd.merge(df_7_y, df_8_y, on=['Distinct_Result'])

    df_7_y = df_7_y.drop(['index'], axis=1)

    df_7_y.to_csv('Distinct_Result_Shelf.csv',index = False,header=True)

    df_7_y_2 = pd.read_csv("Distinct_Result_Shelf.csv")

    df_7_x_2 = df_7_y_2.groupby("N°_of_Shelf_i")['N°_of_Shelf_i'].max()
    df_7_x_2 = pd.DataFrame({'N°_of_Shelf_i':df_7_x_2.index, 'Shelf_classification_prev':df_7_x_2.values})
    df_7_x_2 = df_7_x_2.sort_values(by='Shelf_classification_prev', ascending=True)
    df_7_x_2 = df_7_x_2.reset_index()
    df_7_x_2['Shelf_classification'] = range(1, 1 + len(df_7_x_2))


    num_count = df_7_x_2['Shelf_classification'].count()

    def get_shelf_letter_classification(shelf_letter):
        
        i = -1

        dataframe = pd.read_csv("artifice.csv")
        artifice = dataframe['artifice'].values[0]

        while(i<artifice-1):
            i = i+1
        
        if (artifice<(num_count)):
            dataframe = pd.read_csv("artifice.csv")
            dataframe = dataframe.drop(['artifice'], axis=1)
            dataframe.insert(0,"artifice",[artifice+1],False)
            dataframe.to_csv("artifice.csv",index = False,header=True)
        
        return chr(i+65)

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice.csv",index = False,header=True)

    df_7_x_2['Shelf_classification_letter'] = df_7_x_2['Shelf_classification'].apply(get_shelf_letter_classification)

    df_7_y_2 = pd.merge(df_7_y_2, df_7_x_2, on=['N°_of_Shelf_i'])
    df_7_y_2 = df_7_y_2.drop(['index','Shelf_classification_prev','Shelf_classification'], axis=1)
    df_7_y_2.to_csv('Distinct_Result_Shelf.csv',index = False,header=True)

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice.csv",index = False,header=True)

    dff = pd.read_csv("Item_classification.csv") 
    Item_classication = dff['Item_classification'].values[0]
    
    if Item_classication == "Pareto Analysis":
        dataframe = pd.read_csv("Multi_Pareto.csv") 
        dataframe_2 = dataframe[['Company_Brand', 'Multi_Pareto_Classification']]
        dataframe_3 = pd.read_csv("Multi_Pareto_Shelf.csv") 
        dataframe_4 = pd.merge(dataframe_3, dataframe_2, on=['Company_Brand'])
        dataframe_4['Classification'] = dataframe_4['Multi_Pareto_Classification'] 
        dataframe_5 = dataframe_4[['Company_Brand', 'N°_of_Shelf_i', 'Classification']]
        dataframe_5.to_csv('Set_up_2.csv',index = False,header=True) 
        
    elif Item_classication == "Distinct Category":
        dataframe = pd.read_csv("Distinct_Result_Shelf.csv") 
        dataframe['Classification'] = dataframe['Shelf_classification_letter']
        dataframe_2 = dataframe[['Company_Brand', 'N°_of_Shelf_i', 'Classification']]
        dataframe_2.to_csv('Set_up_2.csv',index = False,header=True) 
    
    df = pd.read_csv("Set_up_2.csv")
    df_2= df.groupby("N°_of_Shelf_i")['N°_of_Shelf_i'].count()
    df_2 = pd.DataFrame({'N°_of_Shelf_i':df_2.index, 'N°_of_Shelf_i_Count':df_2.values})

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice.csv",index = False,header=True)

    empty_list = []

    def get_classification(classification):
        
        i = -1

        dataframe = pd.read_csv("artifice.csv")
        artifice = dataframe['artifice'].values[0]
        while(i<artifice-1):
            i = i+1
                
            dataframe = pd.read_csv("artifice.csv")
            dataframe = dataframe.drop(['artifice'], axis=1)
            dataframe.insert(0,"artifice",[artifice+1],False)
            dataframe.to_csv("artifice.csv",index = False,header=True)
        
        r = np.round(np.random.rand(),1)
        g = np.round(np.random.rand(),1)
        b = np.round(np.random.rand(),1)        

        color=[r,g,b]

        empty_list.append(color)

        return i

    s={'artifice':[1]}
    dataframe = pd.DataFrame(s)
    dataframe.to_csv("artifice.csv",index = False,header=True)

    df_2['N°_of_Shelf_i_Classification'] = df_2['N°_of_Shelf_i'].apply(get_classification)
    df = pd.merge(df, df_2, on=['N°_of_Shelf_i'])
    
    print(empty_list)

    color_categories = df['N°_of_Shelf_i_Classification'].astype(int)
    color_categories = df['N°_of_Shelf_i_Classification'].tolist()
    color_categories = np.array(color_categories)

    color_map = np.array(empty_list) 

    # #create basic bar plot
    
    fig, ax = plt.subplots()
    ax.barh(df['Company_Brand'], df['N°_of_Shelf_i'], color=color_map[color_categories])
    
    plt.xlabel("Areas")
    plt.ylabel("Brands")


    # giving title to the plot
    plt.title("Product Classification")

    #specify axis colors
    ax.tick_params(axis='y', colors=color1)
    

    labels = df['N°_of_Shelf_i'].astype(int)
    labels = labels.tolist()
    plt.xticks(df['N°_of_Shelf_i'], labels)

    #display Pareto chart
    plt.show()


button_2 = customtkinter.CTkButton(widgets_frame, text="Visualize", command=chart)
button_2.grid(row=10, column=0, padx=5, pady=5, sticky="nsew")



from subprocess import call

def open_py_file(): 
    call(["python","config_pareto_distinct_ai_total_shelves.py"])

btn = customtkinter.CTkButton(widgets_frame, text="Set up", command=open_py_file)
btn.grid(row=11, column=0, padx=5, pady=5, sticky="nsew")


dataframe_3 = pd.read_csv("floor_artifice.csv")
status_1 = dataframe_3['status'].values[0]

combo_list_1 = ["Floor Auto", "Floor Manual"]

def combobox_callback(choice):
    print("combobox dropdown clicked:", choice)

status_combobox_2 = customtkinter.CTkComboBox(master=widgets_frame,
                                     values=combo_list_1,
                                     command=combobox_callback)
status_combobox_2.grid(row=12, column=0, padx=5, pady=5,  sticky="ew")
status_combobox_2.set(status_1)



def floor_open_py_file(): 
    status = status_combobox_2.get()
    if(status == "Floor Manual"):
        call(["python","config_floor.py"])
    elif(status == "Floor Auto"):
        call(["python","config_floor_2.py"])

    af = pd.DataFrame({'status':[status]})
    af.to_csv('floor_artifice.csv',index = False,header=True)


btn = customtkinter.CTkButton(widgets_frame, text="Floor Set up", command=floor_open_py_file)
btn.grid(row=13, column=0, padx=5, pady=5, sticky="nsew")


def save(): 
    
    from datetime import datetime
    u={'save_as_type':["Normal Multilayer"], 'save_as_time':[datetime.now()]}
    dataframe = pd.DataFrame(u)
    dataframe.to_csv("save_as_type.csv",index = False,header=True)

    import zipfile
    import os
    
    dfx = pd.read_csv("Images_save.csv")  
    thislist = dfx['thislist'].tolist()
    thislist = list(set(thislist))
    
    # Create a zip file
    with zipfile.ZipFile('Normal_Multilayer_Items.zip', 'w') as zip_file:
        # Add each image to the zip file
        for image_path in thislist:
            # Add the image file to the zip file with its original name
            zip_file.write(image_path, os.path.basename(image_path))

    thislist = list(set(thislist))
    thislist.append('Normal_Multilayer_Items.zip')
    df = pd.DataFrame(thislist, columns=['thislist'])
    dfx = pd.read_csv("Images_save.csv")  
    df2x = pd.concat([df,dfx], ignore_index=True) 
    df2x.to_csv('Images_save.csv',index = False,header=True)

    call(["python","Save.py"])


button_2 = customtkinter.CTkButton(widgets_frame, text="Save", command=save)
button_2.grid(row=14, column=0, padx=5, pady=5, sticky="nsew")


button_2 = customtkinter.CTkButton(widgets_frame, text="Exit", command=root.destroy)
button_2.grid(row=15, column=0, padx=5, pady=5, sticky="nsew")

root.mainloop()    