# K means approach  Future implementation with GA for best Weight combination by maximizing the efficiency


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deap import base, creator, tools, algorithms
import random
from multiprocessing import Pool
import pickle
from collections import defaultdict, deque
from multiprocessing import freeze_support  # Add this import

import logging
from functools import partial
import pathlib

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deap import base, creator, tools, algorithms
import random
import pickle
from collections import defaultdict, deque
import multiprocessing as mp
import os

import multiprocessing  # NEW
from multiprocessing import Pool  # NEW

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


dataframe_max_cluster = pd.read_csv("pre_GA_1.csv")
Max_Storage_Locations = dataframe_max_cluster['Sequential_Storage_Location'].max()



def dynamic_storage_allocation_cluster(df, max_repetitions=storage_location_capacity):
    """
    Enhanced version that:
    1. Shows original location for retrievals in New_Storage_Location
    2. Properly tracks and reuses freed slots
    3. Maintains 20-item limit per location
    """
    # df = pd.read_csv("pre_Shelf_Cluster.csv")
    # print(df)
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
    # print(df)

    return df, location_counts



import pandas as pd
from collections import defaultdict

def track_storage_counts_cluster(df):
    """
    Counts New_Storage_Location values separately:
    - Increases count by 1 when Artifice_1 == 1
    - Decreases count by 1 when Artifice_1 == -1
    """
    df = pd.read_csv("pre_Shelf_Cluster.csv")
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


def evaluate_individual(individual, data_files=("pre_GA.csv", "pre_GA_1.csv", "pre_GA_2.csv")):
    """Evaluation function that works with parallel processing"""
    try:

        # Convert to absolute paths to ensure workers can find files
        data_files = [str(pathlib.Path(f).absolute()) for f in data_files]
        
        df = pd.read_csv(data_files[0])
        df_2 = pd.read_csv(data_files[1]) 
        total_distance_sequential = df_2['Sequential_Distance'].sum()
        count_rows = df_2['Storing'].count()
        efficiency_sequential = (count_rows * 36) / (total_distance_sequential)
        dataframe3 = pd.read_csv(data_files[2])   
        
        df['Product_ID'] = df['ID_order']

        # Verify required columns exist
        required_cols = ['Income', 'Time_priority', 'Count']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            # raise KeyError(f"Missing columns: {missing}")
            return (0,)

        weights = {'Income': individual[0],
                  'Time_priority': individual[1],
                  'Count': individual[2]}
        k = int(individual[3])
        # Validate cluster count
        k = int(individual[3])
        if k < 1 or k > Max_Storage_Locations:  # Use your defined maximum
            return (0,)
       
        # Get features as numpy array
        features = df[required_cols]
        # print(features)
        
        # Apply weights (element-wise multiplication)
        weighted_data = features * weights
        # print(weighted_data)

        # Perform clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(weighted_data)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data) + 1
        
        # Simulate storage allocation
        temp_df = df
        temp_df['Shelf_Cluster'] = clusters
        # print(temp_df)
        temp_df = temp_df[['Shelf_Cluster','Product_ID']]
        merged_df = pd.merge(df_2, temp_df, on='Product_ID', how='left')
        # merged_df['Shelf_Cluster'] = merged_df['Shelf_Cluster'].fillna(1)
        merged_df.to_csv("pre_Shelf_Cluster.csv",index = False,header=True)
        # print(merged_df)

        # Calculate efficiency
        updated_df, _ = dynamic_storage_allocation_cluster(merged_df)
        # print(updated_df)
        updated_df.to_csv("pre_Shelf_Cluster.csv",index = False,header=True)
        
        updated_df = track_storage_counts_cluster(updated_df)
        updated_df.to_csv("pre_Shelf_Cluster.csv",index = False,header=True)

        dataframe3['Fixed_Storage_Location_Cluster'] = dataframe3['ID_Total']
        dataframe3 = dataframe3[['Distance','Fixed_Storage_Location_Cluster']].copy() 
        df_merged_1_Cluster = pd.merge(updated_df, dataframe3, on='Fixed_Storage_Location_Cluster', how='left')
        updated_df['Distance_Cluster'] = df_merged_1_Cluster['Distance']  
        total_distance = updated_df['Distance_Cluster'].sum()
        count_rows = updated_df['Storing'].count()
        efficiency = (count_rows * 36) / (total_distance)
        print(efficiency)
        # logging.basicConfig(filename='ga.log', level=logging.INFO)
        # logging.info(f"Efficiency: {efficiency}")
        with open('ga.log', 'a') as f:  # 'a' for append mode
            f.write(f"Efficiency: {efficiency}\n")
        
        if efficiency > efficiency_sequential:
            return (efficiency,)  # Higher fitness for better solutions
        else:
            return (efficiency * 0.1,)  # Penalize worse solutions (adjust weight)
            
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        # logging.error(f"Evaluation failed: {str(e)}")
        return (0,)


# ========== GENETIC ALGORITHM SETUP ==========
def setup_ga():
    """Initialize GA components"""
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    evaluator = partial(evaluate_individual, 
                      data_files=("pre_GA.csv", "pre_GA_1.csv", "pre_GA_2.csv"))
    toolbox.register("evaluate", evaluator)
    # toolbox.register("evaluate", lambda ind: evaluate_individual(ind, df.copy(), df_2.copy(), dataframe3.copy()))
    # Key Step: Parallel evaluation
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    return toolbox, pool

def create_individual():
    """Create a candidate solution: [w1, w2, w3, K]"""
    weights = np.random.dirichlet(np.ones(3)).tolist()
    k = random.randint(2, Max_Storage_Locations)
    # print(k)
    return weights + [k]

_evaluation_log = []


def _evaluate_and_log(individual, original_evaluate):
    """Standalone function that can be pickled"""
    fitness = original_evaluate(individual)
    _evaluation_log.append((list(individual), fitness[0]))
    return fitness

def optimize_parameters(toolbox, pop_size=80, generations=10):
    """Run GA optimization with verified best solution"""
    global _evaluation_log
    _evaluation_log = []  # Reset log
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    # Store original evaluate function
    original_evaluate = toolbox.evaluate
    
    try:
        # Create a partial function with the original evaluate
        from functools import partial
        evaluate_wrapper = partial(_evaluate_and_log, original_evaluate=original_evaluate)
        toolbox.register("evaluate", evaluate_wrapper)
        
        # Run optimization
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2,
                                     ngen=generations, stats=stats,
                                     halloffame=hof, verbose=True)
        
        # Verification
        if _evaluation_log:
            best_from_log = max(_evaluation_log, key=lambda x: x[1])
            best_from_hof = (list(hof[0]), hof[0].fitness.values[0])
            
            if best_from_log[1] > best_from_hof[1]:
                print(f"Found better solution in log! "
                      f"Log max: {best_from_log[1]:.4f} vs Hof: {best_from_hof[1]:.4f}")
                best_ind = creator.Individual(best_from_log[0])
                best_ind.fitness.values = (best_from_log[1],)
                return best_ind, log
        
        return hof[0], log
        
    finally:
        # Restore original evaluate function
        toolbox.register("evaluate", original_evaluate)

        
def optimize_parameters(toolbox, pop_size=80, generations=10):
    """Run GA optimization"""
    # Moved pool creation inside the main block
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    # With this:
    def get_fitness_values(individual):
        return individual.fitness.values

    stats = tools.Statistics(get_fitness_values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2,
                                  ngen=generations, stats=stats, 
                                  halloffame=hof, verbose=True)
    
    return hof[0], log



##########################################################################

def solution_to_df(best_solution, csv_path='best_solution.csv'):
    """
    Convert DEAP individual to DataFrame and save as CSV
    Returns:
        pd.DataFrame: DataFrame containing the solution details
    """
    # Create the DataFrame
    solution_df = pd.DataFrame({
        'Parameter': ['Income_weight', 'Time_weight', 'Count_weight', 'Clusters_K'],
        'Value': best_solution[:4],
        'Efficiency': [best_solution.fitness.values[0]]*4  # Repeat efficiency for each row
    })
    
    # Save to CSV with additional metadata
    with open(csv_path, 'w') as f:
        f.write(f"# Best Solution from Genetic Algorithm\n")
        f.write(f"# Generated at: {pd.Timestamp.now()}\n")
        solution_df.to_csv(f, index=False)
    
    print(f"Saved best solution to {csv_path}")
    return solution_df

##########################################################################


# ========== CORE FUNCTIONS ==========
def preprocess_data(df):
    """Enhanced version that creates all required columns"""
    df = df.copy()
    
    # 1. Create Time_priority
    df['Time_start'] = pd.to_datetime(df['Time_start'])
    df['Time_finish'] = pd.to_datetime(df['Time_finish'])
    df['Time_spent'] = (df['Time_finish'] - df['Time_start']).dt.total_seconds()
    df['Time_priority'] = 1 / (1 + df['Time_spent'])  # Inverse relationship
    
    # 2. Create Count (frequency per brand)
    brand_counts = df['Company_Brand'].value_counts().reset_index()
    brand_counts.columns = ['Company_Brand', 'Count']
    df = pd.merge(df, brand_counts, on='Company_Brand')
    
    return df

# def create_weighted_features(df, weights):
#     """Apply customizable weights to features"""
#     weighted_df = df[list(weights.keys())].copy()
#     for feature, weight in weights.items():
#         weighted_df[feature] = (weighted_df[feature] - weighted_df[feature].min()) / \
#                               (weighted_df[feature].max() - weighted_df[feature].min()) * weight
#     return weighted_df

def perform_clustering(data, n_clusters):
    """Run K-means clustering"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data) + 1  # 1-indexed
    print(clusters)
    return clusters, kmeans


def final_optimum_clustering(data_files=("pre_GA.csv", "pre_GA_1.csv","pre_GA_2.csv")):

    # Convert to absolute paths to ensure workers can find files
    data_files = [str(pathlib.Path(f).absolute()) for f in data_files]
    
    df = pd.read_csv(data_files[0])
    df_2 = pd.read_csv(data_files[1])   
    dataframe3 = pd.read_csv(data_files[2])   
    
    df['Product_ID'] = df['ID_order']

    # Verify required columns exist
    required_cols = ['Income', 'Time_priority', 'Count']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        # raise KeyError(f"Missing columns: {missing}")
        return (0,)


    best_df = pd.read_csv('best_solution.csv', comment='#')  # Skip metadata lines

    # Convert to weights dictionary and k value
    weights = {
        'Income': best_df[best_df['Parameter'] == 'Income_weight']['Value'].values[0],
        'Time_priority': best_df[best_df['Parameter'] == 'Time_weight']['Value'].values[0],
        'Count': best_df[best_df['Parameter'] == 'Count_weight']['Value'].values[0]
    }
    k = int(best_df[best_df['Parameter'] == 'Clusters_K']['Value'].values[0])

    print(weights)
    print(k)

    # Get features as numpy array
    features = df[required_cols]
    # print(features)
    
    # Apply weights (element-wise multiplication)
    weighted_data = features * weights
    weighted_data.to_csv("optimum_k_weights.csv",index = False,header=True)
    # print(weighted_data)

    # Perform clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(weighted_data)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data) + 1
    
    # Simulate storage allocation
    temp_df = df
    temp_df['Shelf_Cluster'] = clusters
    # print(temp_df)
    temp_df = temp_df[['Shelf_Cluster','Product_ID']]
    merged_df = pd.merge(df_2, temp_df, on='Product_ID', how='left')
    # merged_df['Shelf_Cluster'] = merged_df['Shelf_Cluster'].fillna(1)
    merged_df.to_csv("pre_Shelf_Cluster.csv",index = False,header=True)
    print(merged_df)

    # Calculate efficiency
    updated_df, _ = dynamic_storage_allocation_cluster(merged_df)
    # print(updated_df)
    updated_df.to_csv("pre_Shelf_Cluster.csv",index = False,header=True)
    
    updated_df = track_storage_counts_cluster(updated_df)
    updated_df.to_csv("pre_Shelf_Cluster.csv",index = False,header=True)

    dataframe3['Fixed_Storage_Location_Cluster'] = dataframe3['ID_Total']
    dataframe3 = dataframe3[['Distance','Fixed_Storage_Location_Cluster']].copy() 
    df_merged_1_Cluster = pd.merge(updated_df, dataframe3, on='Fixed_Storage_Location_Cluster', how='left')
    updated_df['Distance_Cluster'] = df_merged_1_Cluster['Distance']  
    total_distance = updated_df['Distance_Cluster'].sum()
    count_rows = updated_df['Storing'].count()
    efficiency = (count_rows * 36) / (total_distance)
    print(efficiency)

    return updated_df


def plot_3d_clusters(df, features, clusters, weights):
    """3D plot of clusters"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        df[features[0]], 
        df[features[1]], 
        df[features[2]], 
        c=clusters, 
        cmap='viridis',
        s=50,
        alpha=0.6
    )
    
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    plt.title(f"Optimal Cluster Visualization\nWeights: {weights}")
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig('optimal_clusters_3d.png', dpi=300)
    plt.show()


def plot_filtered_ga_evolution(log, save_path='ga_evolution_filtered.png'):
    """Plots only generations with successful evaluations"""
    
    # Filter out failed generations (where max fitness == 0)
    valid_generations = [entry for entry in log if entry['max'] > 0]
    
    if not valid_generations:
        print("No valid generations to plot!")
        return
    
    # Extract data from filtered log
    generations = [entry['gen'] for entry in valid_generations]
    max_fitness = [entry['max'] for entry in valid_generations]
    avg_fitness = [entry['avg'] for entry in valid_generations]
    
    # Find best generation
    best_idx = np.argmax(max_fitness)
    best_gen = generations[best_idx]
    best_fitness = max_fitness[best_idx]
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Plot successful evaluations only
    plt.plot(generations, max_fitness, 'b-o', 
             linewidth=2, 
             markersize=8,
             label='Best Fitness (Valid)')
    
    plt.plot(generations, avg_fitness, 'g--s', 
             linewidth=1.5,
             markersize=6,
             label='Avg Fitness (Valid)')
    
    # Highlight best solution
    plt.scatter(best_gen, best_fitness, 
                c='red', 
                s=200,
                edgecolor='black',
                zorder=10,
                label=f'Optimal Solution (Gen {best_gen})')
    
    plt.axvline(x=best_gen, color='red', linestyle=':', alpha=0.5)
    
    # Formatting
    plt.title('GA Optimization Progress (Valid Evaluations Only)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save and show
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

    print(f"Plotted {len(valid_generations)}/{len(log)} valid generations")


# ========== MAIN EXECUTION ==========

def main():
    # Freeze support for Windows (if needed)
    multiprocessing.freeze_support()
    # Load and preprocess data
    df = pd.read_csv("Static_Training_Dataset_2.csv", sep=',')
    df = preprocess_data(df)

    df.to_csv("pre_GA.csv",index = False,header=True)
    df_2 = pd.read_csv("pre_GA_1.csv")
    dataframe3 = pd.read_csv("pre_GA_2.csv")

    # Verify columns
    required = ['Income', 'Time_priority', 'Count']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Setup GA (Parallel)
    print("=== Starting Genetic Algorithm (Parallel) ===")
    # NEW: Initialize GA with parallel support
    toolbox, pool = setup_ga()  # Get pool object
    
    # Run optimization
    best_solution, log = optimize_parameters(toolbox)
    
    # Process results
    if best_solution:
        OPTIMAL_WEIGHTS = {
            'Income': best_solution[0],
            'Time_priority': best_solution[1],
            'Count': best_solution[2]
        }

        OPTIMAL_K = int(best_solution[3])
        
        print(f"\n=== Optimal Parameters ===")
        print(f"Weights: {OPTIMAL_WEIGHTS}")
        print(f"Cluster Count: {OPTIMAL_K}")
        
        # Visualization
        plot_filtered_ga_evolution(log)
        
        solution_df = solution_to_df(best_solution)
        # weighted_data = create_weighted_features(df, OPTIMAL_WEIGHTS)


        df = final_optimum_clustering() 

        weighted_data = pd.read_csv("optimum_k_weights.csv")

        plot_3d_clusters(weighted_data, list(OPTIMAL_WEIGHTS.keys()), 
                        df['Shelf_Cluster'], OPTIMAL_WEIGHTS)
        
        # Save results
        with open('ga_optimization_results.pkl', 'wb') as f:
            pickle.dump({
                'weights': OPTIMAL_WEIGHTS,
                'k': OPTIMAL_K,
                'log': log
            }, f)
    
    
    # Cleanup
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()