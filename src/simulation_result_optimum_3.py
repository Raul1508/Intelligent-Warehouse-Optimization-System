import pandas as pd
import pickle

import tkinter as tk
from tkinter import ttk, PhotoImage
import cv2
import numpy as np
import pandas as pd

import customtkinter
from typing import Union
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from datetime import datetime

from collections import defaultdict, deque



df = pd.read_csv("Static_Training_Dataset_2.csv",sep = ',')

df['Time_start'] = pd.to_datetime(df['Time_start'])
df['Time_finish'] = pd.to_datetime(df['Time_finish'])

df_2_artifice = pd.DataFrame({'Time': pd.concat([df['Time_start'], df['Time_finish']], ignore_index=True)})
df_2 = pd.DataFrame({'Product_ID': pd.concat([df['ID_order'], df['ID_order']], ignore_index=True)})

df_2['Time'] = df_2_artifice['Time']


df_2['Company_Brand'] = pd.concat([df['Company_Brand'], df['Company_Brand']], ignore_index=True)
df_2['Storing'] = df['ID_order']
df_2['Storing']= df_2['Storing'].fillna(-1)

df_2.loc[(df_2['Storing'] > 0), 'Artifice_1'] = 1
df_2.loc[(df_2['Storing'] < 0), 'Artifice_1'] = -1

df_2 = df_2.sort_values(by='Time', ascending=True)
df_2 = df_2.reset_index()
        

df_2['KARDEX'] = df_2['Artifice_1'].cumsum()

Max_items = df_2['KARDEX'].max()

print(Max_items)


prod_size = 100
dataframe3 = pd.read_csv("GUI_part4_distance.csv")

width_p = dataframe3['width_p'].max()
long_p = dataframe3['long_p'].max()

if (long_p >width_p):
    size = long_p

else:
    size = width_p

print(size)

size = size*4.5

import math


dataframe_k = pd.read_csv("floor_config.csv")

#Limit weight per floor
K_levels = dataframe_k['levels'].values[0]

dataframe_warehouse = pd.read_csv("GUI_product_Database.csv")
height_warehouse = dataframe_warehouse['Layout_height'].max()
floor_height = dataframe_warehouse['floor_height'].max()
p_height = dataframe_warehouse['p_height'].max()
max_levels = height_warehouse // (floor_height + p_height*1.1)

K_levels = max_levels


storage_locations = math.ceil(prod_size*Max_items/size/K_levels)

print(storage_locations)

storage_location_capacity = int(size/prod_size)*K_levels

print(storage_location_capacity)



#################################################################


# Distances for every Storage Location

dataframe3 = pd.read_csv("GUI_part4_distance.csv")
pos_long_p_x = dataframe3[dataframe3['ID']==1]['pos_long_p'].max()
pos_width_p_x = dataframe3[dataframe3['ID']==1]['pos_width_p'].max()

if width_p < long_p:

    dataframe3.loc[dataframe3['ID'] == 1, 'Distance'] = (dataframe3['pos_width_p'] + dataframe3['long_p']/2).astype(int)
    dataframe3.loc[(dataframe3['ID'] > 1)&(pos_width_p_x < dataframe3['pos_width_p'])&(pos_long_p_x == dataframe3['pos_long_p']), 'Distance'] = (dataframe3['pos_width_p'] + dataframe3['long_p'] + dataframe3['looseness_corridor_l'] + (dataframe3['long_p'])/2).astype(int)
    dataframe3.loc[(dataframe3['ID'] > 1)&((pos_width_p_x >= dataframe3['pos_width_p'])|(pos_long_p_x != dataframe3['pos_long_p'])), 'Distance'] = (dataframe3['pos_width_p'] + (dataframe3['layout_long'] - dataframe3['pos_long_p'] - dataframe3['long_p']) + dataframe3['long_p']/2).astype(int)
    
else:
    
    dataframe3.loc[dataframe3['ID'] == 1, 'Distance'] = (dataframe3['pos_width_p'] + dataframe3['long_p'] + dataframe3['looseness_corridor_l'] + dataframe3['width_p']/2).astype(int)
    dataframe3.loc[(dataframe3['ID'] > 1)&(pos_width_p_x < dataframe3['pos_width_p'])&(pos_long_p_x == dataframe3['pos_long_p']), 'Distance'] = (dataframe3['pos_width_p'] + dataframe3['long_p'] + dataframe3['width_p']/2 + dataframe3['looseness_corridor_l']).astype(int)
    dataframe3.loc[(dataframe3['ID'] > 1)&((pos_width_p_x >= dataframe3['pos_width_p'])|(pos_long_p_x != dataframe3['pos_long_p'])), 'Distance'] = (dataframe3['pos_width_p'] + (dataframe3['layout_long'] - dataframe3['pos_long_p'] - dataframe3['long_p']) + dataframe3['width_p']/2).astype(int)


################################################################


# Sequential Storage Approach


def sequential_storage_allocation(df, max_repetitions=storage_location_capacity):
    """
    Random storage approach version:
    1. Stores products in first available location (sequential order)
    2. Retrievals decrease count by 1
    3. Maintains max capacity per location
    4. Reuses freed slots
    """
    df = df.copy()
    location_counts = defaultdict(int)
    available_slots = defaultdict(deque)
    product_locations = {}  # {product_id: storage_location}
    new_locations = []
    current_location = 1  # Start from location 1
    
    for idx, row in df.iterrows():
        product_id = f"P{row['Product_ID']:03d}"
        operation = row['Artifice_1']
        
        if operation == -1:  # RETRIEVAL OPERATION
            if product_id in product_locations:
                freed_loc = product_locations[product_id]
                available_slots[freed_loc].append(product_id)
                location_counts[freed_loc] -= 1
                del product_locations[product_id]
                new_locations.append(freed_loc)
            else:
                # For unfound products, use last used location
                last_loc = new_locations[-1] if new_locations else 1
                location_counts[last_loc] = max(0, location_counts[last_loc] - 1)
                new_locations.append(last_loc)
            continue
        
        # STORAGE OPERATION (+1)
        target_loc = None
        
        # Check available slots first
        for loc in sorted(available_slots.keys()):
            if available_slots[loc]:
                target_loc = loc
                break
                
        # If no freed slots, find next available location sequentially
        if target_loc is None:
            loc = current_location
            while True:
                if location_counts.get(loc, 0) < max_repetitions:
                    target_loc = loc
                    current_location = loc  # Move to next location for future searches
                    break
                loc += 1
                # Reset to location 1 if we reach max location
                if loc > max(location_counts.keys(), default=1) + 1:
                    loc = 1
        
        # Use freed slot if available
        if available_slots.get(target_loc):
            _ = available_slots[target_loc].popleft()
        
        # Assign to location
        location_counts[target_loc] += 1
        product_locations[product_id] = target_loc
        new_locations.append(target_loc)
    
    df['Sequential_Storage_Location'] = new_locations
    return df, location_counts


def track_sequential_storage_counts(df):
    """
    Counts New_Storage_Location values separately:
    - Increases count by 1 when Artifice_1 == 1
    - Decreases count by 1 when Artifice_1 == -1
    """
    df = df.copy()
    location_counts = defaultdict(int)
    count_history = []
    
    for idx, row in df.iterrows():
        loc = row['Sequential_Storage_Location']
        operation = row['Artifice_1']
        
        if pd.isna(loc):  # Skip if no location assigned
            count_history.append(None)
            continue
            
        if operation == 1:
            location_counts[loc] += 1
        elif operation == -1:
            location_counts[loc] = max(0, location_counts[loc] - 1)  # Prevent negative counts
        
        count_history.append(location_counts[loc])
    
    df['Sequential_Count'] = count_history
    return df


result_df, _ = sequential_storage_allocation(df_2)
df_2 = track_sequential_storage_counts(result_df)      

Optimum_Storage_Locations = df_2['Sequential_Storage_Location'].max()

# # Calculate cumulative sum of column 'E'
# df_2['Repetitions_prev'] = df_2['Artifice_1'].cumsum()

# # Apply conditions
# df_2['Repetitions'] = df_2['Repetitions_prev'].apply(lambda x: x if x <= storage_location_capacity else x % storage_location_capacity)
# df_2.loc[df_2['Repetitions'] == 0, 'Repetitions'] = storage_location_capacity


# df_2['Artifice_3'] = 0
# df_2.loc[(df_2['Artifice_1'] == -1) & (df_2['Repetitions'] % storage_location_capacity == 0), 'Artifice_3'] = -1
# df_2.loc[(df_2['Artifice_1'] == 1) & (df_2['Repetitions'] % storage_location_capacity == 1), 'Artifice_3'] = 1

# df_2['Random_Storage_Location'] =  df_2['Artifice_3'].cumsum()
# df_2.loc[df_2['Random_Storage_Location'] == 0, 'Random_Storage_Location'] = 1


dataframe3['Sequential_Storage_Location'] = dataframe3['ID_Total'] 

df_merged_2 = pd.merge(df_2, dataframe3, on='Sequential_Storage_Location', how='left')

df_2['Sequential_Distance'] = df_merged_2['Distance']

df_2.to_csv("pre_GA_1.csv",index = False,header=True)
dataframe3.to_csv("pre_GA_2.csv",index = False,header=True)


#################################################################


# Random Storage Approach

import random
import numpy as np

def random_storage_allocation(df, max_repetitions=storage_location_capacity):  # Hardcoded for clarity
    """
    Strict Capacity Random Storage:
    - Enforces max_repetitions at assignment time
    - Verifies capacity before assignment
    - Atomic location selection
    """
    df = df.copy()
    location_counts = defaultdict(int)
    available_slots = defaultdict(deque)
    product_locations = {}
    new_locations = []
    all_locations = set(range(1, Optimum_Storage_Locations + 1))  # Pre-defined locations
    
    for idx, row in df.iterrows():
        product_id = f"P{row['Product_ID']:03d}"
        operation = row['Artifice_1']
        
        if operation == -1:  # RETRIEVAL
            if product_id in product_locations:
                freed_loc = product_locations[product_id]
                available_slots[freed_loc].append(product_id)
                location_counts[freed_loc] -= 1
                del product_locations[product_id]
                new_locations.append(freed_loc)
            continue
        
        # STORAGE OPERATION - STRICT CAPACITY ENFORCEMENT
        valid_locs = [
            loc for loc in all_locations 
            if location_counts[loc] < max_repetitions
        ]
        
        # Prioritize freed slots but verify capacity
        freed_locs = [
            loc for loc in available_slots 
            if available_slots[loc] and location_counts[loc] < max_repetitions
        ]
        
        # Combine candidates
        candidates = freed_locs + [
            loc for loc in valid_locs 
            if loc not in freed_locs
        ]
        
        if not candidates:
            # Emergency expansion (shouldn't happen with proper Optimum_Storage_Locations)
            new_loc = max(all_locations) + 1
            all_locations.add(new_loc)
            candidates = [new_loc]
        
        # Atomic selection and assignment
        target_loc = random.choice(candidates)
        location_counts[target_loc] += 1
        
        # Consume freed slot if available
        if available_slots.get(target_loc):
            _ = available_slots[target_loc].popleft()
        
        product_locations[product_id] = target_loc
        new_locations.append(target_loc)
    
    df['Random_Storage_Location'] = new_locations
    return df, location_counts
    


def track_random_storage_counts(df):
    """
    Counts New_Storage_Location values separately:
    - Increases count by 1 when Artifice_1 == 1
    - Decreases count by 1 when Artifice_1 == -1
    """
    df = df.copy()
    location_counts = defaultdict(int)
    count_history = []
    
    for idx, row in df.iterrows():
        loc = row['Random_Storage_Location']
        operation = row['Artifice_1']
        
        if pd.isna(loc):  # Skip if no location assigned
            count_history.append(None)
            continue
            
        if operation == 1:
            location_counts[loc] += 1
        elif operation == -1:
            location_counts[loc] = max(0, location_counts[loc] - 1)  # Prevent negative counts
        
        count_history.append(location_counts[loc])
    
    df['Random_Count'] = count_history
    return df



result_df, _ = random_storage_allocation(df_2)
df_2 = track_random_storage_counts(result_df)  

dataframe3['Random_Storage_Location'] = dataframe3['ID_Total'] 

df_merged_3 = pd.merge(df_2, dataframe3, on='Random_Storage_Location', how='left')

df_2['Random_Distance'] = df_merged_3['Distance']



#################################################################


# Multi Pareto Set Up:


def set_up():

    layout_num_max = Optimum_Storage_Locations
    
    k={'Company_Brand':[],'N°_of_Shelf_i':[],'Classification':[],'layout_num':[]}
    dataframe3 = pd.DataFrame(k)
    dataframe3.to_csv("Set_up.csv",index = False,header=True)

    i=0

    while( i < layout_num_max):
        
        i = i +1 

        from datetime import datetime

        df = pd.read_csv("Static_Training_Dataset_2.csv",sep = ',')

        df['Time_start'] = pd.to_datetime(df['Time_start'])
        df['Time_finish'] = pd.to_datetime(df['Time_finish'])

        df['Time_spent'] = df['Time_finish'] - df['Time_start'] 

        df_21= df.groupby("Company_Brand")['Time_spent'].sum()

        df_21 = pd.DataFrame({'Company_Brand':df_21.index, 'Sum_Time':df_21.values})

        df_21 = df_21.sort_values(by='Sum_Time', ascending=False)
        Total_Sum_Time = df_21['Sum_Time'].sum()
        df_21['%Time'] = df_21['Sum_Time']/Total_Sum_Time
        df_21['%Acum_Time'] = df_21['%Time'].cumsum(axis=0)
        
        df_21.loc[df_21['%Acum_Time'] <= 0.75, 'Pareto_Time'] = 'C' 
        df_21.loc[(df_21['%Acum_Time'] <= 0.90) & (df_21['%Acum_Time'] > 0.75), 'Pareto_Time'] = 'B'
        df_21.loc[(df_21['%Acum_Time'] <= 1) & (df_21['%Acum_Time'] > 0.90), 'Pareto_Time'] = 'A' 

        df_21.loc[df_21['Pareto_Time'] == 'A', 'Pareto_Time_num'] = 3
        df_21.loc[df_21['Pareto_Time'] == 'B', 'Pareto_Time_num'] = 2
        df_21.loc[df_21['Pareto_Time'] == 'C', 'Pareto_Time_num'] = 1

        df_21.to_csv('Pareto_Time.csv',index = False,header=True)

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


        df_4 = pd.merge(df_21, df_3, on=['Company_Brand'])
        df_4 = df_4[['Company_Brand','Sum_Time','Count']]
        df_4['Time_per_spent'] = df_4['Sum_Time']/df_4['Count']
        df_4 = df_4.sort_values(by='Time_per_spent', ascending=False)
        Total_Time_per_spent = df_4['Time_per_spent'].sum()
        df_4['%Time_per_spent'] = df_4['Time_per_spent']/Total_Time_per_spent
        df_4.drop(['Sum_Time', 'Count'], axis=1)
        df_4['%Acum_Time_per_spent'] = df_4['%Time_per_spent'].cumsum(axis=0)
        
        df_4.loc[df_4['%Acum_Time_per_spent'] <= 0.75, 'Pareto_Time_per_spent'] = 'C' 
        df_4.loc[(df_4['%Acum_Time_per_spent'] <= 0.90) & (df_4['%Acum_Time_per_spent'] > 0.75), 'Pareto_Time_per_spent'] = 'B'
        df_4.loc[(df_4['%Acum_Time_per_spent'] <= 1) & (df_4['%Acum_Time_per_spent'] > 0.90), 'Pareto_Time_per_spent'] = 'A' 

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


        df_6 = pd.merge(df_21, df_3, on=['Company_Brand'])
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


        Shelfs_max = layout_num_max


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
        dataframe3 = pd.DataFrame(s)
        dataframe3.to_csv("artifice.csv",index = False,header=True)

        s={'artifice':[1]}
        dataframe3 = pd.DataFrame(s)
        dataframe3.to_csv("artifice_2.csv",index = False,header=True)

        first_row = df_8['N°_of_Shelf_f'].values[0]
        first_multi_pareto_result = df_8['Multi_Pareto_Result'].values[0]
        num_of_results = df_8['Multi_Pareto_Result'].count()

        def get_shelf_i(shelf_i):
                y = -1

                dataframe3 = pd.read_csv("artifice.csv")
                artifice = dataframe3['artifice'].values[0]

                while(y<artifice-1):
                    y = y+1

                if (shelf_i == first_row)&(first_multi_pareto_result == df_8['Multi_Pareto_Result'].values[y]):
                    dataframe3 = pd.read_csv("artifice.csv")
                    dataframe3 = dataframe3.drop(['artifice'], axis=1)
                    dataframe3.insert(0,"artifice",[artifice+1],False)
                    dataframe3.to_csv("artifice.csv",index = False,header=True)
                    return 1
                # if (shelf_i == first_row)|(first_multi_pareto_result == df_8['Multi_Pareto_Result'].values[y]):
                #     if(first_multi_pareto_result == df_8['Multi_Pareto_Result'].values[y]):
                #         dataframe3 = pd.read_csv("artifice.csv")
                #         dataframe3 = dataframe3.drop(['artifice'], axis=1)
                #         dataframe3.insert(0,"artifice",[artifice+1],False)
                #         dataframe3.to_csv("artifice.csv",index = False,header=True)
                #         return 1
                #     else:
                #         s={'artifice':[1]}
                #         dataframe3 = pd.DataFrame(s)
                #         dataframe3.to_csv("artifice.csv",index = False,header=True)
                #         return df_8['N°_of_Shelf_f'].values[y]
                else:

                    i = -1

                    dataframe3 = pd.read_csv("artifice_2.csv")
                    artifice = dataframe3['artifice'].values[0]
                    while(i<artifice-1):
                        i = i+1
                            
                        dataframe3 = pd.read_csv("artifice_2.csv")
                        dataframe3 = dataframe3.drop(['artifice'], axis=1)
                        dataframe3.insert(0,"artifice",[artifice+1],False)
                        dataframe3.to_csv("artifice_2.csv",index = False,header=True)
                    
                    if (df_8['Multi_Pareto_Result'].values[i]==df_8['Multi_Pareto_Result'].values[i-1]):
                        return df_8['N°_of_Shelf_f'].values[i-1]        
                    else:        
                        return df_8['N°_of_Shelf_f'].values[i]
                

        s={'artifice':[1]}
        dataframe3 = pd.DataFrame(s)
        dataframe3.to_csv("artifice.csv",index = False,header=True)

        s={'artifice':[1]}
        dataframe3 = pd.DataFrame(s)
        dataframe3.to_csv("artifice_2.csv",index = False,header=True)

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

                dataframe3 = pd.read_csv("artifice.csv")
                artifice = dataframe3['artifice'].values[0]

                while(y<artifice-1):
                    y = y+1

                if (shelf_i == first_row)&(first_distinct_result == df_8_y['Distinct_Result'].values[y]) :
                    dataframe3 = pd.read_csv("artifice.csv")
                    dataframe3 = dataframe3.drop(['artifice'], axis=1)
                    dataframe3.insert(0,"artifice",[artifice+1],False)
                    dataframe3.to_csv("artifice.csv",index = False,header=True)
                    return 1
                    
                else:
                    i = -1

                    dataframe3 = pd.read_csv("artifice_2.csv")
                    artifice = dataframe3['artifice'].values[0]
                    
                    while(i<artifice-1):
                        i = i+1
                        dataframe3 = pd.read_csv("artifice_2.csv")
                        dataframe3 = dataframe3.drop(['artifice'], axis=1)
                        dataframe3.insert(0,"artifice",[artifice+1],False)
                        dataframe3.to_csv("artifice_2.csv",index = False,header=True)
                    
                    # data = pd.read_csv("weights.csv")
                    # weight_time = data[data['weight_type']=='time weight']['weight'].max() 
                    # weight_frequency = data[data['weight_type']=='frequency weight']['weight'].max()
                    
                    
                    if (df_8_y['Distinct_Result'].values[i]==df_8_y['Distinct_Result'].values[i-1]):
                        return df_8_y['N°_of_Shelf_f'].values[i-1]        
                    else: 
                        return df_8_y['N°_of_Shelf_f'].values[i]

                    
                    # if(weight_time > 0)|(weight_frequency > 0):
                    #     if (df_8_y['Distinct_Result'].values[i]==df_8_y['Distinct_Result'].values[i+1]):
                    #         return df_8_y['N°_of_Shelf_f'].values[i]        
                    #     else: 
                    #         return df_8_y['N°_of_Shelf_f'].values[i+1]
                    # else:
                    #     if (df_8_y['Distinct_Result'].values[i]==df_8_y['Distinct_Result'].values[i-1]):
                    #         return df_8_y['N°_of_Shelf_f'].values[i-1]        
                    #     else: 
                    #         return df_8_y['N°_of_Shelf_f'].values[i]
                        
                

        s={'artifice':[1]}
        dataframe3 = pd.DataFrame(s)
        dataframe3.to_csv("artifice.csv",index = False,header=True)

        s={'artifice':[1]}
        dataframe3 = pd.DataFrame(s)
        dataframe3.to_csv("artifice_2.csv",index = False,header=True)

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

            dataframe3 = pd.read_csv("artifice.csv")
            artifice = dataframe3['artifice'].values[0]

            while(i<artifice-1):
                i = i+1
            
            if (artifice<(num_count)):
                dataframe3 = pd.read_csv("artifice.csv")
                dataframe3 = dataframe3.drop(['artifice'], axis=1)
                dataframe3.insert(0,"artifice",[artifice+1],False)
                dataframe3.to_csv("artifice.csv",index = False,header=True)
            
            return chr(i+65)

        s={'artifice':[1]}
        dataframe3 = pd.DataFrame(s)
        dataframe3.to_csv("artifice.csv",index = False,header=True)

        df_7_x_2['Shelf_classification_letter'] = df_7_x_2['Shelf_classification'].apply(get_shelf_letter_classification)

        df_7_y_2 = pd.merge(df_7_y_2, df_7_x_2, on=['N°_of_Shelf_i'])
        df_7_y_2 = df_7_y_2.drop(['index','Shelf_classification_prev','Shelf_classification'], axis=1)
        df_7_y_2.to_csv('Distinct_Result_Shelf.csv',index = False,header=True)

        s={'artifice':[1]}
        dataframe3 = pd.DataFrame(s)
        dataframe3.to_csv("artifice.csv",index = False,header=True)

        dataframe3 = pd.read_csv("Multi_Pareto.csv") 
        dataframe_2 = dataframe3[['Company_Brand', 'Multi_Pareto_Classification']]
        dataframe_3 = pd.read_csv("Multi_Pareto_Shelf.csv") 
        dataframe_4 = pd.merge(dataframe_3, dataframe_2, on=['Company_Brand'])
        dataframe_4['Classification'] = dataframe_4['Multi_Pareto_Classification'] 
        dataframe_5 = dataframe_4[['Company_Brand', 'N°_of_Shelf_i', 'Classification']]
        dataframe_5['layout_num'] = i
        # dataframe_5['layout_num'] = pd.Series([layout_num for x in range(len(dataframe_5.index))])  
        df = pd.read_csv("Set_up.csv")  
        df2 = pd.concat([dataframe_5,df], ignore_index=True) 
        df2.to_csv('Set_up.csv',index = False,header=True) 
             

trigger_multi_pareto = set_up() 

#################################################################


df_set_up = pd.read_csv("Set_up_2.csv")

df_merged = pd.merge(df_2, df_set_up, on='Company_Brand', how='left')

df_2['Storage_Location'] = df_merged['N°_of_Shelf_i']


#####################################################


def dynamic_storage_allocation_cluster(df, max_repetitions=storage_location_capacity):
    """
    Enhanced version that:
    1. Shows original location for retrievals in New_Storage_Location
    2. Properly tracks and reuses freed slots
    3. Maintains 20-item limit per location
    """
    df = df.copy()
    location_counts = defaultdict(int)
    available_slots = defaultdict(deque)
    product_locations = {}  # Tracks current location of each product
    new_locations = []
    
    for idx, row in df.iterrows():
        product_id = f"P{row['Product_ID']:03d}"  # Format as P001, P002, etc.
        operation = row['Artifice_1']
        original_loc = row['Shelf_Cluster']
        
        if operation == -1:  # RETRIEVAL OPERATION
            if product_id in product_locations:
                freed_loc = product_locations[product_id]
                available_slots[freed_loc].append(product_id)
                location_counts[freed_loc] -= 1
                del product_locations[product_id]
                new_locations.append(freed_loc)  # Show where product was stored
            else:
                last_loc = new_locations[-1]
                location_counts[last_loc] = max(0, location_counts[last_loc] - 1)
                new_locations.append(last_loc)

            continue
        
        # STORAGE OPERATION (+1)
        target_loc = None
        
        # 1. First try original location if available
        if (location_counts.get(original_loc, 0) < max_repetitions and 
            not available_slots.get(original_loc)):
            target_loc = original_loc
        
        # 2. Check for freed slots in original location
        elif available_slots.get(original_loc):
            target_loc = original_loc
        
        # 3. Find next sequential available location
        else:
            loc = original_loc
            while True:
                if (location_counts.get(loc, 0) < max_repetitions or 
                    available_slots.get(loc)):
                    target_loc = loc
                    break
                loc += 1
        
        # Use freed slot if available
        if available_slots.get(target_loc):
            _ = available_slots[target_loc].popleft()  # Remove from available
        
        # Assign to location
        location_counts[target_loc] += 1
        product_locations[product_id] = target_loc
        new_locations.append(target_loc)
    
    df['Fixed_Storage_Location_Cluster'] = new_locations
    return df, location_counts



import pandas as pd
from collections import defaultdict

def track_storage_counts_cluster(df):
    """
    Counts New_Storage_Location values separately:
    - Increases count by 1 when Artifice_1 == 1
    - Decreases count by 1 when Artifice_1 == -1
    """
    df = df.copy()
    location_counts = defaultdict(int)
    count_history = []
    
    for idx, row in df.iterrows():
        loc = row['Fixed_Storage_Location_Cluster']
        operation = row['Artifice_1']
        
        if pd.isna(loc):  # Skip if no location assigned
            count_history.append(None)
            continue
            
        if operation == 1:
            location_counts[loc] += 1
        elif operation == -1:
            location_counts[loc] = max(0, location_counts[loc] - 1)  # Prevent negative counts
        
        count_history.append(location_counts[loc])
    
    df['Location_Count_Cluster'] = count_history
    return df


#####################################################


# K means approach  Future implementation with GA for best Weight combination by maximizing the efficiency


import subprocess
import pandas as pd
import os

# Run the script and wait for completion
subprocess.run(["python", "GA_K_clustering.py"], check=True)

print("\nOptimization complete! Results saved.")

if os.path.exists("GA_Best_K_and_weights.csv"):
    dataframe_cluster = pd.read_csv("GA_Best_K_and_weights.csv")
    
    # Convert 'Company_Brand' to string in both DataFrames
    df_2['Company_Brand'] = df_2['Company_Brand'].astype(str)
    dataframe_cluster['Company_Brand'] = dataframe_cluster['Company_Brand'].astype(str)
    
    # Now perform the merge
    df_merged_cluster = pd.merge(df_2, dataframe_cluster, on='Company_Brand', how='left')
    df_2['Shelf_Cluster'] = df_merged_cluster['Shelf_Cluster']
else:
    print("Error: CSV file not created within timeout period")


#####################################################


def dynamic_storage_allocation(df, max_repetitions=storage_location_capacity):
    """
    Enhanced version that:
    1. Shows original location for retrievals in New_Storage_Location
    2. Properly tracks and reuses freed slots
    3. Maintains 20-item limit per location
    """
    df = df.copy()
    location_counts = defaultdict(int)
    available_slots = defaultdict(deque)
    product_locations = {}  # Tracks current location of each product
    new_locations = []
    
    for idx, row in df.iterrows():
        product_id = f"P{row['Product_ID']:03d}"  # Format as P001, P002, etc.
        operation = row['Artifice_1']
        original_loc = row['Storage_Location']
        
        if operation == -1:  # RETRIEVAL OPERATION
            if product_id in product_locations:
                freed_loc = product_locations[product_id]
                available_slots[freed_loc].append(product_id)
                location_counts[freed_loc] -= 1
                del product_locations[product_id]
                new_locations.append(freed_loc)  # Show where product was stored
            else:
                last_loc = new_locations[-1]
                location_counts[last_loc] = max(0, location_counts[last_loc] - 1)
                new_locations.append(last_loc)

            continue
        
        # STORAGE OPERATION (+1)
        target_loc = None
        
        # 1. First try original location if available
        if (location_counts.get(original_loc, 0) < max_repetitions and 
            not available_slots.get(original_loc)):
            target_loc = original_loc
        
        # 2. Check for freed slots in original location
        elif available_slots.get(original_loc):
            target_loc = original_loc
        
        # 3. Find next sequential available location
        else:
            loc = original_loc
            while True:
                if (location_counts.get(loc, 0) < max_repetitions or 
                    available_slots.get(loc)):
                    target_loc = loc
                    break
                loc += 1
        
        # Use freed slot if available
        if available_slots.get(target_loc):
            _ = available_slots[target_loc].popleft()  # Remove from available
        
        # Assign to location
        location_counts[target_loc] += 1
        product_locations[product_id] = target_loc
        new_locations.append(target_loc)
    
    df['Fixed_Storage_Location'] = new_locations
    return df, location_counts



import pandas as pd
from collections import defaultdict

def track_storage_counts(df):
    """
    Counts New_Storage_Location values separately:
    - Increases count by 1 when Artifice_1 == 1
    - Decreases count by 1 when Artifice_1 == -1
    """
    df = df.copy()
    location_counts = defaultdict(int)
    count_history = []
    
    for idx, row in df.iterrows():
        loc = row['Fixed_Storage_Location']
        operation = row['Artifice_1']
        
        if pd.isna(loc):  # Skip if no location assigned
            count_history.append(None)
            continue
            
        if operation == 1:
            location_counts[loc] += 1
        elif operation == -1:
            location_counts[loc] = max(0, location_counts[loc] - 1)  # Prevent negative counts
        
        count_history.append(location_counts[loc])
    
    df['Location_Count'] = count_history
    return df



#####################################################



# df_2 = df[['Storage_Location', 'Artifice_1']]
# df_2['Product_ID'] = range(1, len(df_2) + 1)

updated_df, counts = dynamic_storage_allocation(df_2)
updated_df_2 = track_storage_counts(updated_df)
updated_df_3, _ = dynamic_storage_allocation_cluster(updated_df_2)  # Unpack correctly
updated_df_4 = track_storage_counts_cluster(updated_df_3)

df_2 = updated_df_4


#########################################################################

# dataframe3 distance from here for Multi Pareto and Kmeans clustering

dataframe3['Fixed_Storage_Location'] = dataframe3['ID_Total'] 
print(dataframe3)

dataframe3['Fixed_Storage_Location_Cluster'] = dataframe3['ID_Total'] 
print(dataframe3)


# dataframe3.to_csv('Distances_for_first_paper.csv',index = False,header=True)

df_merged_1 = pd.merge(df_2, dataframe3, on='Fixed_Storage_Location', how='left')

df_2['Multi_Pareto_Distance'] = df_merged_1['Distance']


df_merged_1_Cluster = pd.merge(df_2, dataframe3, on='Fixed_Storage_Location_Cluster', how='left')

df_2['Distance_Cluster'] = df_merged_1_Cluster['Distance']



#######################################################################


Sequential_Distance = df_2['Sequential_Distance'].sum() # in
Multi_Pareto_Distance = df_2['Multi_Pareto_Distance'].sum() # in
Random_Distance = df_2['Random_Distance'].sum() # in
Cluster_Distance = df_2['Distance_Cluster'].sum() # in
count_id = df_2['Storing'].count()

Efficiency_of_Sequential_Approach = count_id*36/Sequential_Distance # product/yd
Efficiency_of_Multi_Pareto_Approach = count_id*36/Multi_Pareto_Distance # product/yd
Efficiency_of_Random_Approach = count_id*36/Random_Distance # product/yd
Efficiency_of_Cluster = count_id*36/Cluster_Distance # product/yd


Efficiency_improvement_sequential = (Random_Distance - Sequential_Distance) / Random_Distance   

Efficiency_improvement_multi_pareto = (Random_Distance - Multi_Pareto_Distance) / Random_Distance 

Efficiency_improvement_cluster = (Random_Distance - Cluster_Distance) / Random_Distance 




# Space Utilization Metrics

Storage_density = df_2['KARDEX'].max() / ((dataframe3['layout_width'].max())*(dataframe3['layout_long'].max())) 

max_efficiency = max(Efficiency_of_Sequential_Approach, Efficiency_of_Multi_Pareto_Approach, Efficiency_of_Random_Approach, Efficiency_of_Cluster)

fill_rates = df_2.copy()

if max_efficiency == Efficiency_of_Sequential_Approach:
    fill_rates['item_count'] = df_2['Sequential_Count']
    fill_rates['storage_location'] = df_2['Sequential_Storage_Location']
elif max_efficiency == Efficiency_of_Multi_Pareto_Approach:
    fill_rates['item_count'] = df_2['Location_Count'] 
    fill_rates['storage_location'] = df_2['Fixed_Storage_Location']
elif max_efficiency == Efficiency_of_Random_Approach:
    fill_rates['item_count'] = df_2['Random_Count']
    fill_rates['storage_location'] = df_2['Random_Storage_Location']
elif max_efficiency == Efficiency_of_Cluster:
    fill_rates['item_count'] = df_2['Location_Count_Cluster']
    fill_rates['storage_location'] = df_2['Fixed_Storage_Location_Cluster']

# Calculate fill rate for each location
fill_rates['fill_rate'] = fill_rates['item_count'] / storage_location_capacity  # max_repetitions = 20 (or your value)
fill_rates['dead_space'] = 1 - fill_rates['fill_rate']


fill_rates.loc[(fill_rates['Artifice_1'] == -1), 'storing'] = 0
fill_rates.loc[(fill_rates['Artifice_1'] == 1), 'storing'] = 1


# Add additional stats
fill_rates['utilization_status'] = fill_rates['fill_rate'].apply(
    lambda x: 'Overutilized' if x > 0.9 else 
              'Optimal' if x > 0.7 else 
              'Underutilized'
)


# Aggregate stats by location
summary = fill_rates.groupby('storage_location').agg({
    'storing': 'sum',                     # Total items stored
    'fill_rate': ['mean', 'std'],         # Fill rate stats
    'dead_space': ['mean', 'std'],        # Dead space stats
    'utilization_status': lambda x: x.mode()[0],  # Most common status 
    'item_count': 'mean'        # Peak and average traffic
}).reset_index()

# Flatten column names correctly
summary.columns = [
    'storage_location',
    'total_items',              # Sum of storing
    'location_fill_rate',       # mean fill_rate
    'fill_rate_std_dev',        # std fill_rate
    'dead_space_rate',          # mean dead_space
    'dead_space_std_dev',       # std dead_space
    'utilization_status',       # mode status
    'Avg_Traffic'               # mean item_count
]


print(summary['utilization_status'].dtype)
print(summary['Avg_Traffic'].dtype)

summary['Capacity_Location_Traffic'] = storage_location_capacity

# Calculate Congestion_Index
summary['Congestion_Index'] = summary['Avg_Traffic'] / summary['Capacity_Location_Traffic']

# Add overall warehouse metrics
warehouse_avg_fill = fill_rates['fill_rate'].mean()
warehouse_std_dev = fill_rates['fill_rate'].std()

summary.to_csv("summary_data.csv",index = False,header=True)



# According to: ISO 11228-1:2003 applies to moderate walking speed, i.e. 0,5 m/s to 1,0 m/sec on a horizontal level surface.

Average_Walking_speed = (0.5)/(0.0254) # in/s
Sequential_Time = Sequential_Distance/(Average_Walking_speed*60*60) # hours
Multi_Pareto_Time = Multi_Pareto_Distance/(Average_Walking_speed*60*60) # hours
Random_Time = Random_Distance/(Average_Walking_speed*60*60) # hours
Cluster_Time = Cluster_Distance/(Average_Walking_speed*60*60) # hours


Time_Reduction_Sequential = Random_Time - Sequential_Time # hours

Time_Improvement_Sequential = (Random_Time-Sequential_Time)/Random_Time


Time_Reduction_Multi_Pareto = Random_Time - Multi_Pareto_Time # hours

Time_Improvement_Multi_Pareto = (Random_Time-Multi_Pareto_Time)/Random_Time


Time_Reduction_Cluster = Random_Time - Cluster_Time # hours

Time_Improvement_Cluster = (Random_Time-Cluster_Time)/Random_Time


# Average_salary_of_a_warehouse_worker = 34 # yuan/hour

# Money_saved_by_the_system = Average_salary_of_a_warehouse_worker*Time_Reduction # yuan in 2000 orders

# Money_saved_by_the_system_cluster = Average_salary_of_a_warehouse_worker*Time_Reduction_Cluster # yuan in 2000 orders

# print(Time_Reduction)
# print(Time_Improvement)
# print(Money_saved_by_the_system)



dataframe3 = pd.read_csv("GUI_part4_distance.csv")
Layers = dataframe3['layout_num'].max()
Layers = int(Layers)

import os
import pandas as pd

# Specify the file path
file_path = "Result_Analysis_4_storage_approaches.csv"

# Check if the file already exists
if not os.path.exists(file_path):
    # Create DataFrame (or use your existing DataFrame)

    d = {'Layers':[],'Storage_Location_Width':[],'Storage_Location_Long':[],
    'Sequential_Distance':[],'Multi_Pareto_Distance':[],'Random_Distance':[],'Cluster_Distance':[],
    'Efficiency_of_Sequential_Approach':[],'Efficiency_of_Multi_Pareto_Approach':[],
    'Efficiency_of_Random_Approach':[],'Efficiency_of_Cluster':[],'Efficiency_improvement_sequential':[],'Efficiency_improvement_multi_pareto':[],
    'Efficiency_improvement_cluster':[],'Storage_density':[],'warehouse_avg_fill':[],
    'warehouse_std_dev':[],'Sequential_Time':[],'Multi_Pareto_Time':[],
    'Random_Time':[],'Cluster_Time':[],'Time_Reduction_Sequential':[],'Time_Improvement_Sequential':[],'Time_Reduction_Multi_Pareto':[],
    'Time_Improvement_Multi_Pareto':[],'Time_Reduction_Cluster':[],'Time_Improvement_Cluster':[]}
    
    df_4 = pd.DataFrame(d)
    # Save DataFrame to CSV file
    df_4.to_csv("Result_Analysis_4_storage_approaches.csv", index=False)
    print("CSV file created successfully.")
else:
    print("CSV file already exists.")

Multi_Pareto_Storage_Locations = df_2['Fixed_Storage_Location'].max()
Storage_Location_Width = width_p 
Storage_Location_Long = long_p

Random_Storage_Locations = df_2['Random_Storage_Location'].max()
print(Random_Storage_Locations)

af = pd.DataFrame({'Layers':[Layers],'Storage_Location_Width':[Storage_Location_Width],'Storage_Location_Long':[Storage_Location_Long],
    'Sequential_Distance':[Sequential_Distance],'Multi_Pareto_Distance':[Multi_Pareto_Distance],'Random_Distance':[Random_Distance],'Cluster_Distance':[Cluster_Distance],
    'Efficiency_of_Sequential_Approach':[Efficiency_of_Sequential_Approach],'Efficiency_of_Multi_Pareto_Approach':[Efficiency_of_Multi_Pareto_Approach],
    'Efficiency_of_Random_Approach':[Efficiency_of_Random_Approach],'Efficiency_of_Cluster':[Efficiency_of_Cluster],
    'Efficiency_improvement_sequential':[Efficiency_improvement_sequential],'Efficiency_improvement_multi_pareto':[Efficiency_improvement_multi_pareto],
    'Efficiency_improvement_cluster':[Efficiency_improvement_cluster],'Storage_density':[Storage_density],'warehouse_avg_fill':[warehouse_avg_fill],
    'warehouse_std_dev':[warehouse_std_dev],'Sequential_Time':[Sequential_Time],'Multi_Pareto_Time':[Multi_Pareto_Time],
    'Random_Time':[Random_Time],'Cluster_Time':[Cluster_Time],'Time_Reduction_Sequential':[Time_Reduction_Sequential],'Time_Improvement_Sequential':[Time_Improvement_Sequential],
    'Time_Reduction_Multi_Pareto':[Time_Reduction_Multi_Pareto],
    'Time_Improvement_Multi_Pareto':[Time_Improvement_Multi_Pareto],'Time_Reduction_Cluster':[Time_Reduction_Cluster],'Time_Improvement_Cluster':[Time_Improvement_Cluster]})

print(af)
df_4 = pd.read_csv("Result_Analysis_4_storage_approaches.csv",sep = r'[\t , ;]')  
print(df_4)
df_3 = pd.concat([af,df_4], ignore_index=True) 

df_3 = df_3.sort_values(by='Random_Time', ascending=True)
df_3 = df_3.reset_index(drop=True)

df_3.to_csv('Result_Analysis_4_storage_approaches.csv',index = False,header=True)


#From excel:

# df_2['Artifice_2'] = 0
# df_2.loc[(df_2['Artifice_1'] == -1) & (df_2['Cumulative_Sum'] % 27 == 0), 'Artifice_2'] = -1
# df_2.loc[(df_2['Artifice_1'] == 1) & (df_2['Cumulative_Sum'] % 27 == 1), 'Artifice_2'] = 1

# # Calculate the cumulative sum based on condition
# df_2['Prev_Storage_Location'] = df_2.groupby('Storage_Location')['Artifice_2'].cumsum()

# # Apply the formula
# df_2['Fixed_Storage_Location'] = df_2['Storage_Location'] + df_2['Prev_Storage_Location'] -1



print(df_2)

df_2.to_csv("simulation_result.csv",index = False,header=True)

##########################################################################

# 3D plot


import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Load your data
fill_rates['events'] = range(1, len(fill_rates) + 1)

# Create 3D plot with smooth lines (alternative to spline)
fig = go.Figure(data=[
    go.Scatter3d(
        x=fill_rates['events'],
        y=fill_rates['storage_location'],
        z=fill_rates['item_count'],  # Using index as event length proxy
        mode='lines+markers',
        line=dict(
            color='#FF6B6B',
            width=6,
            # For 3D smoothing, we'll use a different approach:
            # 1. Interpolate your data first for smoother lines
            # 2. Or increase data density
        ),
        marker=dict(
            size=8,
            color=fill_rates['storage_location'],
            colorscale='Viridis',
            opacity=0.9,
            line=dict(width=2, color='white'),
            colorbar=dict(title='Count Intensity')
        ),
        customdata=df_2,
        hovertemplate=
            "<b>Location</b>: %{x}<br>"
            "<b>Count</b>: %{y}<br>"
            "<b>Sequence</b>: %{z}<br>"
            "<extra></extra>"
    )
])

# For smoother lines in 3D, pre-process your data:
from scipy import interpolate

# Create interpolation function
tck, u = interpolate.splprep([
    fill_rates['events'],
    fill_rates['storage_location'],
    fill_rates['item_count']
], s=0)

# Create denser interpolated points
u_new = np.linspace(0, 1, 100)  # 10x density
x_new, y_new, z_new = interpolate.splev(u_new, tck)

# Add smoothed trace
fig.add_trace(go.Scatter3d(
    x=x_new,
    y=y_new,
    z=z_new,
    mode='lines',
    line=dict(
        color='rgba(255,107,107,0.7)',
        width=8
    ),
    showlegend=False
))

# Rest of your layout configuration remains the same
fig.update_layout(
    scene=dict(
        xaxis=dict(
            title='Event Sequence →',
            gridcolor='rgba(100,100,100,0.5)',
            tickfont=dict(size=16, color='white')  # Axis title here
        ),
        yaxis=dict(
            title='← Storage Location',
            gridcolor='rgba(100,100,100,0.5)',
            tickfont=dict(size=16, color='white')  # Axis title here
        ),
        zaxis=dict(
            title='↑ Repetition Count',
            gridcolor='rgba(100,100,100,0.5)',
            tickfont=dict(size=16, color='white')  # Axis title here
        ),
        bgcolor='rgba(10,10,20,1)',
        camera=dict(
            eye=dict(x=1.5, y=-1.5, z=0.8),
            up=dict(x=0, y=0, z=1)
        )
    ),
    title=dict(  # This is where the main figure title goes
        text="<b>STORAGE EVOLUTION ANALYSIS</b><br>"
             "<span style='font-size:12px'>Location vs. Count vs. Sequence</span>",
        font=dict(size=24, color="#FFFFFF"),
        x=0.05,
        y=0.95
    ),
    scene_bgcolor='rgba(10,10,20,1)',
    paper_bgcolor='rgba(10,10,20,1)',
    height=900
)

fig.show()





####################################################################################

# 2D plot


import plotly.graph_objects as go
import pandas as pd
import numpy as np
# from colours import mcolors

# Load your data
# df_2 = pd.read_csv("simulation_result.csv")  # Replace with your actual data loading



# Create custom vibrant color palette
location_colors = {
    1: '#FF355E',  # Radical Red
    2: '#FFFF66',  # Laser Lemon
    3: '#00CC99',  # Caribbean Green
    4: '#FF6EFF',  # Shocking Pink
    5: '#5D8AA8',  # Air Force Blue
    6: '#FF9966',  # Atomic Tangerine
    7: '#00FFFF',  # Cyan
    8: '#FF00FF',  # Magenta
    9: '#AAFF00',  # Lime
    10: '#FF007F'  # Rose
}

# Create figure with enhanced effects
fig_2 = go.Figure()

# Add traces for each unique storage location
for location in fill_rates['storage_location'].unique():
    loc_data = fill_rates[fill_rates['storage_location'] == location]
    
    # Main trace
    fig_2.add_trace(go.Scatter(
        x=loc_data['events'],
        y=loc_data['item_count'],
        mode='lines+markers',
        name=f'Location {location}',
        line=dict(
            width=4,
            color=location_colors.get(location, '#FFFFFF'),
            shape='spline',
            smoothing=1.1
        ),
        marker=dict(
            size=10,
            symbol='hexagon',
            opacity=0.9,
            line=dict(width=2, color='white')
        ),
        customdata=np.stack((
            loc_data['storage_location'],
            loc_data['events'],
            loc_data['item_count']
        ), axis=-1),
        hovertemplate=
            "<b>Location %{customdata[0]}</b><br>"
            "Event: %{customdata[1]}<br>"
            "Repetitions: %{customdata[2]}<br>"
            "<extra></extra>"
    ))
    
    # Glow effect trace
    fig_2.add_trace(go.Scatter(
        x=loc_data['events'],
        y=loc_data['item_count'],
        mode='lines',
        showlegend=False,
        line=dict(
            width=18,
            color=location_colors.get(location, '#FFFFFF'),
            shape='spline',
            smoothing=1.1
        ),
        opacity=0.12
    ))

fig_2.update_layout(
    title=dict(
        text="<b>STORAGE LOCATION PERFORMANCE METRICS</b><br>"
             "<span style='font-size:14px;color:#AAAAAA'>Repetition Patterns Across Events</span>",
        font=dict(family="Arial Black", size=26, color="#FFFFFF"),
        x=0.03,
        y=0.97,
        xanchor='left'
    ),
    xaxis=dict(
        title="<b>Event Sequence</b>",
        gridcolor='rgba(80,80,80,0.3)',
        tickfont=dict(size=18, color='white'),
        linecolor='white',
        mirror=True,
        zeroline=False,
        showspikes=True,
        spikethickness=1,
        spikedash="dot"
    ),
    yaxis=dict(
        title="<b>Repetition Count</b>",
        gridcolor='rgba(80,80,80,0.3)',
        tickfont=dict(size=18, color='white'),
        linecolor='white',
        mirror=True,
        zeroline=False,
        showspikes=True
    ),
    plot_bgcolor='rgba(15,15,25,1)',
    paper_bgcolor='rgba(10,10,20,1)',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=14, color="white"),
        bgcolor="rgba(0,0,0,0.5)",  # Corrected the property here
        itemsizing='constant'
    ),
    hoverlabel=dict(
        bgcolor="rgba(0,0,0,0.8)",
        font_size=14,
        font_family="Courier New",
        font_color="white",
        bordercolor="rgba(255,255,255,0.5)"
    ),
    height=800,
    width=1200,
    margin=dict(t=120, b=80, l=80, r=40),
    hovermode="x unified"
)

# Add dynamic range selector
fig_2.update_layout(
    xaxis=dict(
        rangeslider=dict(
            visible=True,
            thickness=0.08,
            bgcolor='rgba(50,50,70,0.6)'
        ),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ]),
            bgcolor='rgba(20,20,40,0.8)',
            activecolor='rgba(100,100,150,0.8)',
            font=dict(color='white')
        )
    )
)

# # Add animation controls
# fig_2.update_layout(
#     updatemenus=[dict(
#         type="buttons",
#         buttons=[
#             dict(
#                 label="▶️ Play Animation",  # Button label
#                 method="animate",
#                 args=[None, {"frame": {"duration": 150, "redraw": True}}],
#                 # Adjust button text color using 'color' (valid property)
#                 # font=dict(color='white'),
#                 # Adjust border color and width (valid properties)
#                 # bordercolor='rgba(50,150,50,0.7)',  
#                 # borderwidth=2
#             ),
#             dict(
#                 label="⏹ Stop",  # Button label
#                 method="animate",
#                 args=[[None], {"frame": {"duration": 0, "redraw": True}}],
#                 # Adjust button text color using 'color' (valid property)
#                 # font=dict(color='white'),
#                 # Adjust border color and width (valid properties)
#                 # bordercolor='rgba(150,50,50,0.7)',  
#                 # borderwidth=2
#             )
#         ],
#         direction="left",
#         pad={"r": 10, "t": 10},
#         showactive=True,
#         x=0.1,
#         xanchor="right",
#         y=1.15,
#         yanchor="top"
#     )]
# )



# Create animation frames
frames = []
for i in range(5, len(fill_rates['events'].unique()) + 1, 2):
    frame_range = fill_rates['events'].unique()[:i]
    frames.append(go.Frame(
        data=[go.Scatter(
            x=fill_rates[fill_rates['storage_location'] == loc]['events'].isin(frame_range),
            y=fill_rates[fill_rates['storage_location'] == loc]['item_count'].where(
                fill_rates['events'].isin(frame_range))
        ) for loc in fill_rates['storage_location'].unique()],
        layout=dict(
            xaxis=dict(range=[fill_rates['events'].min(), frame_range[-1]]),
            yaxis=dict(range=[0, fill_rates['item_count'].max() * 1.1])
        )
    ))

fig_2.frames = frames

# Enable advanced features
config = {
    'scrollZoom': True,
    'displayModeBar': True,
    'modeBarButtonsToAdd': [
        'drawline',
        'drawopenpath',
        'drawcircle',
        'drawrect',
        'eraseshape',
        'hoverclosest'
    ],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'storage_location_analysis',
        'scale': 3
    }
}

fig_2.show(config=config)


