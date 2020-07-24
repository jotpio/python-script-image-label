import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from sklearn import preprocessing
from pylab import *
import os.path
import os
import json 
import glob

from tqdm import tqdm, tqdm_notebook
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#### Load sloth json into pandas dataframe
def loadJSONintoDF(filepath):
    assert (os.path.isfile(filepath)) # check if file path exists
    
    loaded_jsons = json.load( open(filepath, 'r') )
    assert (len(loaded_jsons) == 1), f"More than on annotated image was found in json file: {filepath}" #there must be a 1:1 mapping of json-files and images
    annos = loaded_jsons[0]['annotations']
    df = pd.DataFrame(annos)
    return df
#### Load all sloth jsons into array of pandas dataframes
def loadAllJSONSFromPath(datapath):
    assert (os.path.isdir(datapath)) # check if datapath is directory
    json_files = [pos_json for pos_json in os.listdir(datapath) if pos_json.endswith('.json')]
    
    all_df = []
    
    for filename in json_files:
        #df_stats = df_stats.append({'filename' : filename} , ignore_index=True)
        all_df.append(loadJSONintoDF(datapath + filename))
    df_stats = pd.DataFrame(0, index=json_files, columns=['Ntags_f1', 'Ntags_f2', 'Ntags_f3'])
    df_stats['Ntags_f1'] = 0
    df_stats['Ntags_f2'] = 0
    df_stats['Ntags_f3'] = 0
    df_stats['Ntags_total'] = 0
    
    #print(json_files)
    
    return df_stats, all_df, json_files

def calcNumberOfClassesInAnno(df):
    classes = ['fish', 'fish_2', 'fish_3']
    dfout = pd.DataFrame()
    class_ab = [0,0,0]
    
    for i, cla in enumerate(classes):
        class_ab[i] = len( df.loc[df['class'] == cla])
    return class_ab
             
    
# add number of classes of all files to data statistics
def getNumberOfClassesInDFs(df_stats, all_df, json_files):
    for i, df in enumerate(all_df):
        df_nclasses = calcNumberOfClassesInAnno(df)
        
        df_stats.at[json_files[i], 'Ntags_f1'] = df_nclasses[0]
        df_stats.at[json_files[i], 'Ntags_f2'] = df_nclasses[1]
        df_stats.at[json_files[i], 'Ntags_f3'] = df_nclasses[2]
        df_stats.at[json_files[i], 'Ntags_total'] = sum(df_nclasses)
        
    return df_stats

def splitPos(df_stats, all_df, json_files):
    '''
    sloth annotation has xn and yn with
    xn = xhead;xtail
    yn = yhead;ytail
    this fctn. splits the yn and xn to new columns in the dataframe
    '''
    df_allsplit = []

    for df in all_df:
        df_split = df.copy()

        xn = np.array([ np.array(xn.split(';'), dtype=float) for xn in df['xn']])
        df_split['xhead'] = pd.Series( xn[:, 0] )
        df_split['xtail'] = pd.Series( xn[:, 1] )
        yn = np.array([ np.array(yn.split(';'), dtype=float) for yn in df['yn']])
        df_split['yhead'] = pd.Series( yn[:, 0] )
        df_split['ytail'] = pd.Series( yn[:, 1] )

        if 'xn' in df_split.columns and 'yn' in df_split.columns:
            df_split = df_split.drop(['xn', 'yn'], axis = 1)

        # add average position ((pos_head + pos_tail) / 2) column
        df_split['x_av'] = (df_split['xhead'] + df_split['xtail']) / 2
        df_split['y_av'] = (df_split['yhead'] + df_split['ytail']) / 2

        # add length column
        df_split['len'] = np.sqrt(np.power(df_split['xhead'] - df_split['xtail'], 2) + np.power(df_split['yhead']- df_split['ytail'], 2))

        df_split['dir_rad'] = np.arctan2((df_split['yhead']- df_split['ytail']), (df_split['xhead'] - df_split['xtail']))
        #print(df_split.head(1))

        df_allsplit.append(df_split)
    assert(len(df_allsplit) == len(json_files))
    
    return df_allsplit

def calc_neighbors(df_split, number):
    df_pos_av = df_split[['x_av', 'y_av']]

    dist_m = distance_matrix(df_pos_av, df_pos_av)
    df_dist_m = pd.DataFrame(dist_m)
    
    # calculate nnd
    nnd_array = []
    for idx, distances in enumerate(dist_m):
        distances = np.delete(distances, idx) # remove distance to self from list
        nnd = np.min(distances)
        nnd_array.append(nnd)
        
    assert(len(nnd_array) == len(df_split.index)), 'Error in distance calculations!'
    df_split['nnd_px'] = nnd_array
        
    #find neighbours in distance matrix
    avg_length = df_split['len'].mean()
    dist2BL = avg_length * 2
    dist4BL = avg_length * 4
    
    df_class_group_length = df_split.groupby('class')['len'].mean()
    
    #avg BL f1
    avg_lenght_f1 = np.nan
    dist_2_f1_BL = np.nan
    if("fish" in df_class_group_length):
        avg_lenght_f1 = df_class_group_length["fish"]
        dist_2_f1_BL = avg_lenght_f1 * 2 
        
    #avg BL f2
    avg_lenght_f2 = np.nan
    dist_2_f2_BL = np.nan
    df_split.groupby('class')['len'].mean()
    if("fish_2" in df_class_group_length):
        avg_lenght_f2 = df_class_group_length["fish_2"]
        dist_2_f2_BL = avg_lenght_f2 * 2
    
    print(f"{number}:   total avg 2BL: {dist2BL:.5f} , total avg 4BL: {dist4BL:.5f}, f1 avg 2BL: {dist_2_f1_BL}, f2 avg 2BL: {dist_2_f2_BL} ")

    
    
    
    #neighbors in distance total BL2
    df_dist_m_2 = df_dist_m[df_dist_m < dist2BL]
    df_dist_m_2 = df_dist_m_2[df_dist_m_2 != 0]
    assert(df_dist_m_2.shape[0] == df_split.shape[0])  #this should be the same as before

    np_dists = df_dist_m_2.values
    np_neighbor_pairs = np.argwhere(~np.isnan(np_dists))
    #np_neighbors2 = np.split(np_neighbor_pairs[:, 1], np.cumsum(np.unique(np_neighbor_pairs[:, 0], return_counts=True)[1])[:-1]) #https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function

    #setup neighbors total BL2
    np_neighbors2 = []
    for i_fish in range(df_dist_m.shape[0]):
        np_neighbors2.append([])
    #fill neighbors
    for neighbor_pair in np_neighbor_pairs:
        np_neighbors2[neighbor_pair[0]].append(neighbor_pair[1])
        
    assert(len(np_neighbors2) == df_split.shape[0])
    df_split['nb_av2'] = np_neighbors2

   


    #neighbors in distance total BL4
    df_dist_m_4 = df_dist_m[df_dist_m < dist4BL]
    df_dist_m_4 = df_dist_m_4[df_dist_m_4 != 0]
    assert(df_dist_m_4.shape[0] == df_split.shape[0])  #this should be the same as before

    np_dists = df_dist_m_4.values
    np_neighbor_pairs = np.argwhere(~np.isnan(np_dists))
    #np_neighbors4 = np.split(np_neighbor_pairs[:, 1], np.cumsum(np.unique(np_neighbor_pairs[:, 0], return_counts=True)[1])[:-1])    #https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function

    #setup neighbors total BL4
    np_neighbors4 = []
    for i_fish in range(df_dist_m.shape[0]):
        np_neighbors4.append([])
    #fill neighbors
    for neighbor_pair in np_neighbor_pairs:
        np_neighbors4[neighbor_pair[0]].append(neighbor_pair[1])
    
    assert(len(np_neighbors4) == df_split.shape[0])
    df_split['nb_av4'] = np_neighbors4
    
    
    
    
    #neighbors in distance fish1 BL2
    df_dist_m_2f1BL = df_dist_m[df_dist_m < dist_2_f1_BL]
    df_dist_m_2f1BL = df_dist_m_2f1BL[df_dist_m_2f1BL != 0]
    assert(df_dist_m_2f1BL.shape[0] == df_split.shape[0])  #this should be the same as before

    np_dists = df_dist_m_2f1BL.values
    np_neighbor_pairs = np.argwhere(~np.isnan(np_dists))
    #setup neighbors fish1 BL2
    np_neighbors2f1BL = []
    for i_fish in range(df_dist_m.shape[0]):
        np_neighbors2f1BL.append([])
    #fill neighbors
    for neighbor_pair in np_neighbor_pairs:
        np_neighbors2f1BL[neighbor_pair[0]].append(neighbor_pair[1])
        
    assert(len(np_neighbors2f1BL) == df_split.shape[0])
    df_split['nb_f1BL2'] = np_neighbors2f1BL
    
    
    
    
    
    #neighbors in distance fish2 BL2
    
    df_dist_m_2f2BL = df_dist_m[df_dist_m < dist_2_f2_BL]
    df_dist_m_2f2BL = df_dist_m_2f2BL[df_dist_m_2f2BL != 0]
    assert(df_dist_m_2f2BL.shape[0] == df_split.shape[0])  #this should be the same as before
    np_dists = df_dist_m_2f2BL.values
    np_neighbor_pairs = np.argwhere(~np.isnan(np_dists))
    #setup neighbors fish1 BL2
    np_neighbors2f2BL = []
    for i_fish in range(df_dist_m.shape[0]):
        np_neighbors2f2BL.append([])
    #fill neighbors
    for neighbor_pair in np_neighbor_pairs:
        np_neighbors2f2BL[neighbor_pair[0]].append(neighbor_pair[1])

    assert(len(np_neighbors2f2BL) == df_split.shape[0])
    df_split['nb_f2BL2'] = np_neighbors2f2BL
    
    
    
    
    
    return df_split
    
def calc_all_neighbors(df_allsplit, json_files):
    
    #get all numbers
    data_numbers = []
    for json in json_files:
        imgpath, number = get_imgpath_for_json(json)
        data_numbers.append(number)
        
    for i, df_split in enumerate(tqdm_notebook(df_allsplit)):
        calc_neighbors(df_split, data_numbers[i])
    return df_allsplit, data_numbers
        
#### Find path of image linked to json  
# this only works with this format: IMG_XXXX_annotations_al.json and IMG_XXXX.jpg
def get_imgpath_for_json(filename):
    #print(filename)
    number = filename[4:8]
        
    all_imgdir = './data/images/'
    imgfiles = [img_name for img_name in os.listdir(all_imgdir) if img_name.endswith('.jpg')]
    found_file = -1
    
    for name in imgfiles:
        if name[4: -4] == number:
            found_file = name
    assert(found_file != -1)
    imgpath = all_imgdir + found_file
    assert (os.path.isfile(imgpath))
    
    return imgpath, number

def calc_class_neighbors(df_split):
    
    # get classes of neighbors
    neighbor_list_f = []
    neighbor_list_f2 = []
    neighbor_list_f3 = []
    
    np_neighbors_totalBL2 = df_split["nb_av2"].tolist()
    np_neighbors_totalBL4 = df_split["nb_av4"].tolist()
    np_neighbors_totalf1BL2 = df_split["nb_f1BL2"].tolist()
    np_neighbors_totalf2BL2 = df_split["nb_f2BL2"].tolist()


    assert (len(np_neighbors_totalBL2) == df_split.shape[0])
    assert (len(np_neighbors_totalBL4) == df_split.shape[0])
    assert (len(np_neighbors_totalf1BL2) == df_split.shape[0])
    assert (len(np_neighbors_totalf2BL2) == df_split.shape[0])

    #count neighbors for distance total BL2
    for nl in np_neighbors_totalBL2:
        neighbors_f = []
        neighbors_f2 = []
        neighbors_f3 = []

        for n in nl:
            switch = (df_split['class'][n])
            if switch =='fish':
                neighbors_f.append(n)
            elif switch == 'fish_2':
                neighbors_f2.append(n)
            elif switch == 'fish_3':
                neighbors_f3.append(n)

        neighbor_list_f.append(neighbors_f)
        neighbor_list_f2.append(neighbors_f2)
        neighbor_list_f3.append(neighbors_f3)

    # count neighbors
    neighbor_numbers_f = [len(n) for n in neighbor_list_f]
    neighbor_numbers_f2 = [len(n) for n in neighbor_list_f2]
    neighbor_numbers_f3 = [len(n) for n in neighbor_list_f3]

    # add class neighbor list to df
    df_split['#neighbors_av2_f'] = neighbor_numbers_f
    df_split['#neighbors_av2_f2'] = neighbor_numbers_f2
    df_split['#neighbors_av2_f3'] = neighbor_numbers_f3

    #############
    #############

    #count neighbors for distance total BL4
    neighbor_list_av4_f = []
    neighbor_list_av4_f2 = []
    neighbor_list_av4_f3 = []

    for nl in np_neighbors_totalBL4:
        neighbors_av4_f = []
        neighbors_av4_f2 = []
        neighbors_av4_f3 = []

        for n in nl:
            switch = (df_split['class'][n])
            if switch =='fish':
                neighbors_av4_f.append(n)
            elif switch == 'fish_2':
                neighbors_av4_f2.append(n)
            elif switch == 'fish_3':
                neighbors_av4_f3.append(n)

        neighbor_list_av4_f.append(neighbors_av4_f)
        neighbor_list_av4_f2.append(neighbors_av4_f2)
        neighbor_list_av4_f3.append(neighbors_av4_f3)

    # count neighbors
    neighbor_numbers_f = [len(n) for n in neighbor_list_av4_f]
    neighbor_numbers_f2 = [len(n) for n in neighbor_list_av4_f2]
    neighbor_numbers_f3 = [len(n) for n in neighbor_list_av4_f3]

    # add class neighbor list to df
    df_split['#neighbors_av4_f'] = neighbor_numbers_f
    df_split['#neighbors_av4_f2'] = neighbor_numbers_f2
    df_split['#neighbors_av4_f3'] = neighbor_numbers_f3
    
    
    #############
    #############
    
    #count neighbors for distance fish1 BL2
    neighbor_list_f1BL2_f = []
    neighbor_list_f1BL2_f2 = []
    neighbor_list_f1BL2_f3 = []

    for nl in np_neighbors_totalf1BL2:
        neighbors_f1BL2_f = []
        neighbors_f1BL2_f2 = []
        neighbors_f1BL2_f3 = []

        for n in nl:
            switch = (df_split['class'][n])
            if switch =='fish':
                neighbors_f1BL2_f.append(n)
            elif switch == 'fish_2':
                neighbors_f1BL2_f2.append(n)
            elif switch == 'fish_3':
                neighbors_f1BL2_f3.append(n)

        neighbor_list_f1BL2_f.append(neighbors_f1BL2_f)
        neighbor_list_f1BL2_f2.append(neighbors_f1BL2_f2)
        neighbor_list_f1BL2_f3.append(neighbors_f1BL2_f3)

    # count neighbors
    neighbor_numbers_f = [len(n) for n in neighbor_list_f1BL2_f]
    neighbor_numbers_f2 = [len(n) for n in neighbor_list_f1BL2_f2]
    neighbor_numbers_f3 = [len(n) for n in neighbor_list_f1BL2_f3]

    # add class neighbor list to df
    df_split['#neighbors_f1BL2_f'] = neighbor_numbers_f
    df_split['#neighbors_f1BL2_f2'] = neighbor_numbers_f2
    df_split['#neighbors_f1BL2_f3'] = neighbor_numbers_f3
    
    #############
    #############
    
    #count neighbors for distance fish2 BL2
    neighbor_list_f2BL2_f = []
    neighbor_list_f2BL2_f2 = []
    neighbor_list_f2BL2_f3 = []

    for nl in np_neighbors_totalf2BL2:
        neighbors_f2BL2_f = []
        neighbors_f2BL2_f2 = []
        neighbors_f2BL2_f3 = []

        for n in nl:
            switch = (df_split['class'][n])
            if switch =='fish':
                neighbors_f2BL2_f.append(n)
            elif switch == 'fish_2':
                neighbors_f2BL2_f2.append(n)
            elif switch == 'fish_3':
                neighbors_f2BL2_f3.append(n)

        neighbor_list_f2BL2_f.append(neighbors_f2BL2_f)
        neighbor_list_f2BL2_f2.append(neighbors_f2BL2_f2)
        neighbor_list_f2BL2_f3.append(neighbors_f2BL2_f3)

    # count neighbors
    neighbor_numbers_f = [len(n) for n in neighbor_list_f2BL2_f]
    neighbor_numbers_f2 = [len(n) for n in neighbor_list_f2BL2_f2]
    neighbor_numbers_f3 = [len(n) for n in neighbor_list_f2BL2_f3]

    # add class neighbor list to df
    df_split['#neighbors_f2BL2_f'] = neighbor_numbers_f
    df_split['#neighbors_f2BL2_f2'] = neighbor_numbers_f2
    df_split['#neighbors_f2BL2_f3'] = neighbor_numbers_f3

def all_calc_class_neighbours(df_allsplit):
    for df_split in tqdm_notebook(df_allsplit):
        calc_class_neighbors(df_split)
    return df_allsplit

def nb_stats_calculations(df_allsplit, df_stats):
    
    # length, degree, density calculations
    mean_length_f1 = []
    mean_length_f2 = []
    mean_length_f3 = []
            
    
    av2_degree = []
    av2_density = []
    
    mean_nnd = []
    
    for i, df_split in enumerate(df_allsplit):
        lengths_f1 = []
        lengths_f2 = []
        lengths_f3 = []
        
        av2_num_total = df_stats["Ntags_total"][i]
        
        degree_array_BL2 = []
        
        for i in df_split.index:
            c = df_split.at[i, "class"]
            if(c == "fish"):
                lengths_f1.append(df_split.at[i,"len"])
            elif(c == "fish_2"):
                lengths_f2.append(df_split.at[i,"len"])
            elif(c == "fish_3"):
                lengths_f3.append(df_split.at[i,"len"])
            
            # degree of fish
            degree_fish_BL2 = len(df_split.at[i, "nb_av2"])
            degree_array_BL2.append(degree_fish_BL2)
                
        mean_length_f1.append(np.mean(lengths_f1) if len(lengths_f1) > 0 else np.nan)
        mean_length_f2.append(np.mean(lengths_f2) if len(lengths_f2) > 0 else np.nan)
        mean_length_f3.append(np.mean(lengths_f3) if len(lengths_f3) > 0 else np.nan)
        
        mean_degree_BL2 = np.mean(degree_array_BL2)
        av2_degree.append(mean_degree_BL2)
        av2_density.append(mean_degree_BL2/av2_num_total if av2_num_total != 0 else 0)
        
        mean_nnd.append(np.mean(df_split["nnd_px"]))
    
    ##############################################################################
    ################################ Distance BL2 ################################
    ##############################################################################
    
    av2_allf_same = []
    av2_allf2_same = []
    av2_allf3_same = []

    av2_avg_total_n = []
    av2_avg_ff_n = []
    av2_avg_f2f2_n = []
    av2_avg_f3f3_n = []
    
    _2f1BL_allf_same = []
    _2f2BL_allf2_same = []
    _2f1BL_avg_ff_n = []
    _2f2BL_avg_f2f2_n = []
    
    for i, df_split in enumerate(df_allsplit):

        av2_p_f_same = 0
        av2_p_f2_same = 0
        av2_p_f3_same = 0

        av2_avg_f_same = 0
        av2_avg_f2_same = 0
        av2_avg_f3_same = 0
        
        _2f1BL_p_f1_same = 0
        _2f2BL_p_f2_same = 0
        _2f1BL_avg_f1_same = 0
        _2f2BL_avg_f2_same = 0

        num_f = df_stats["Ntags_f1"][i]
        num_f2 = df_stats["Ntags_f2"][i]
        num_f3 = df_stats["Ntags_f3"][i]

        av2_avg_n_number = 0
        
        # calculate average number of neighbors of same class of fish by all neighbors of fish per class and append to list of images
        # also calculate average number of same class neighbors and total neighbors
        for i in df_split.index:
            # avg number of same class neighbors and percentual
            c = df_split.at[i, "class"]
            av2_neighbors_number = len(df_split.at[i, "nb_av2"])
            _2f1BL_neighbors_number = len(df_split.at[i, "nb_f1BL2"])
            _2f2BL_neighbors_number = len(df_split.at[i, "nb_f2BL2"])

            av2_avg_n_number += av2_neighbors_number

            if(av2_neighbors_number == 0):
                continue

            if(c == "fish"):
                av2_p_f_same += df_split.at[i, "#neighbors_av2_f"] / av2_neighbors_number
                av2_avg_f_same += df_split.at[i, "#neighbors_av2_f"]
                
                _2f1BL_p_f1_same += df_split.at[i, "#neighbors_f1BL2_f"] / _2f1BL_neighbors_number
                _2f1BL_avg_f1_same += df_split.at[i, "#neighbors_f1BL2_f"]
                
            elif(c == "fish_2"):
                av2_p_f2_same += df_split.at[i, "#neighbors_av2_f2"] / av2_neighbors_number
                av2_avg_f2_same += df_split.at[i, "#neighbors_av2_f2"]
                
                _2f2BL_p_f2_same += df_split.at[i, "#neighbors_f2BL2_f2"] / _2f2BL_neighbors_number
                _2f2BL_avg_f2_same += df_split.at[i, "#neighbors_f2BL2_f2"]
            elif(c == "fish_3"):
                av2_p_f3_same += df_split.at[i, "#neighbors_av2_f3"] / av2_neighbors_number
                av2_avg_f3_same += df_split.at[i, "#neighbors_av2_f3"]

        av2_allf_same.append(av2_p_f_same/num_f if num_f != 0 else 0)
        av2_allf2_same.append(av2_p_f2_same/num_f2 if num_f2 != 0 else 0)
        av2_allf3_same.append(av2_p_f3_same/num_f3 if num_f3 != 0 else 0)

        av2_avg_total_n.append(av2_avg_n_number/df_split.shape[0] if df_split.shape[0] != 0 else 0)
        av2_avg_ff_n.append(av2_avg_f_same/num_f if num_f != 0 else 0)
        av2_avg_f2f2_n.append(av2_avg_f2_same/num_f2 if num_f2 != 0 else 0)
        av2_avg_f3f3_n.append(av2_avg_f3_same/num_f3 if num_f3 != 0 else 0)
        
        
        _2f1BL_allf_same.append(_2f1BL_p_f1_same/num_f if num_f != 0 else 0)
        _2f2BL_allf2_same.append(_2f2BL_p_f2_same/num_f2 if num_f2 != 0 else 0)
        _2f1BL_avg_ff_n.append(_2f1BL_avg_f1_same/num_f if num_f != 0 else 0)
        _2f2BL_avg_f2f2_n.append(_2f2BL_avg_f2_same/num_f2 if num_f2 != 0 else 0)
        
        

    # Add results to df_stats                          
    
    df_stats["PercSameSpecNeighbors_per2BL_f1"] = av2_allf_same
    df_stats["PercSameSpecNeighbors_per2BL_f2"] = av2_allf2_same
    df_stats["PercSameSpecNeighbors_per2BL_f3"] = av2_allf3_same

    df_stats["NSameSpecNeighbors_per2BL_total"] = av2_avg_total_n
    df_stats["NSameSpecNeighbors_per2BL_f1"] = av2_avg_ff_n
    df_stats["NSameSpecNeighbors_per2BL_f2"] = av2_avg_f2f2_n
    df_stats["NSameSpecNeighbors_per2BL_f3"] = av2_avg_f3f3_n
    
    df_stats["PercSameSpecNeighbors_per2f1BL_f1"] = _2f1BL_allf_same
    df_stats["PercSameSpecNeighbors_per2f2BL_f2"] = _2f2BL_allf2_same

    df_stats["NSameSpecNeighbors_per2f1BL_f1"] = _2f1BL_avg_ff_n
    df_stats["NSameSpecNeighbors_per2f2BL_f2"] = _2f2BL_avg_f2f2_n
    
    df_stats["meanDegree_per2BL_total"] = av2_degree
    df_stats["meanDensity_per2BL_total"] = av2_density
    
    df_stats["meanNND_total"] = mean_nnd
                              
    df_stats["meanBL_px_f1"] = mean_length_f1
    df_stats["meanBL_px_f2"] = mean_length_f2
    df_stats["meanBL_px_f3"] = mean_length_f3

    
    ##############################################################################
    ################################ Distance BL4 ################################
    ##############################################################################
    
    av4_allf_same = []
    av4_allf2_same = []
    av4_allf3_same = []

    av4_avg_total_n = []
    av4_avg_ff_n = []
    av4_avg_f2f2_n = []
    av4_avg_f3f3_n = []

    for i, df_split in enumerate(df_allsplit):

        av4_p_f_same = 0
        av4_p_f2_same = 0
        av4_p_f3_same = 0

        av4_avg_f_same = 0
        av4_avg_f2_same = 0
        av4_avg_f3_same = 0

        av4_num_f = df_stats["Ntags_f1"][i]
        av4_num_f2 = df_stats["Ntags_f2"][i]
        av4_num_f3 = df_stats["Ntags_f3"][i]

        av4_avg_n_len = 0

        # calculate average number of neighbors of same class of fish by all neighbors of fish per class and append to list of images
        # also calculate average number of same class neighbors and total neighbors
        for i in df_split.index:
            #avg number of same class neighbors and percentual
            c = df_split.at[i, "class"]
            av4_neighbors_len = len(df_split.at[i, "nb_av4"])

            av4_avg_n_len += av4_neighbors_len

            if(av4_neighbors_len == 0):
                continue

            if(c == "fish"):
                av4_p_f_same += df_split.at[i, "#neighbors_av4_f"] / av4_neighbors_len
                av4_avg_f_same += df_split.at[i, "#neighbors_av4_f"]
            elif(c == "fish_2"):
                av4_p_f2_same += df_split.at[i, "#neighbors_av4_f2"] / av4_neighbors_len
                av4_avg_f2_same += df_split.at[i, "#neighbors_av4_f2"]
            elif(c == "fish_3"):
                av4_p_f3_same += df_split.at[i, "#neighbors_av4_f3"] / av4_neighbors_len
                av4_avg_f3_same += df_split.at[i, "#neighbors_av4_f3"]

        av4_allf_same.append(av4_p_f_same/av4_num_f if av4_num_f != 0 else 0)
        av4_allf2_same.append(av4_p_f2_same/av4_num_f2 if av4_num_f2 != 0 else 0)
        av4_allf3_same.append(av4_p_f3_same/av4_num_f3 if av4_num_f3 != 0 else 0)

        av4_avg_total_n.append(av4_avg_n_len/df_split.shape[0] if df_split.shape[0] != 0 else 0)
        av4_avg_ff_n.append(av4_avg_f_same/av4_num_f if av4_num_f != 0 else 0)
        av4_avg_f2f2_n.append(av4_avg_f2_same/av4_num_f2 if av4_num_f2 != 0 else 0)
        av4_avg_f3f3_n.append(av4_avg_f3_same/av4_num_f3 if av4_num_f3 != 0 else 0)


    df_stats["PercSameSpecNeighbors_per4BL_f1"] = av4_allf_same
    df_stats["PercSameSpecNeighbors_per4BL_f2"] = av4_allf2_same
    df_stats["PercSameSpecNeighbors_per4BL_f3"] = av4_allf3_same

    df_stats["NSameSpecNeighbors_per4BL_total"] = av4_avg_total_n
    df_stats["NSameSpecNeighbors_per4BL_f1"] = av4_avg_ff_n
    df_stats["NSameSpecNeighbors_per4BL_f2"] = av4_avg_f2f2_n
    df_stats["NSameSpecNeighbors_per4BL_f3"] = av4_avg_f3f3_n

    return df_stats

def calc_polarization(df):
    
    df_head = df[['xhead', 'yhead']]
    df_tail = df[['xtail', 'ytail']]

    df_head = df_head.rename(columns={"xhead": "x", "yhead": "y"})
    df_tail = df_tail.rename(columns={"xtail": "x", "ytail": "y"})

    # center the fish voctors around 0,0
    df_zeroed = df_head - df_tail

    # get unit vectors
    zeroed = df_zeroed.values
    unitv = preprocessing.normalize(zeroed, norm='l2')

    # mean of unit vectors
    mean_uv = np.mean(unitv, axis=0)

    # magnitude of mean = polarization
    pol = np.linalg.norm(mean_uv)

    # add pol error to df_stats
    pol_error = 1/np.sqrt(len(zeroed)) # rough estimate of error
    
    return pol, pol_error

def pop_stats_pol_dir_len(df_allsplit, df_stats):
    all_avg_len_total = []
    all_avg_dir = []
    all_pol = []
    all_pol_error = []
    for idf, df_split in enumerate(tqdm_notebook(df_allsplit)):
        all_avg_len_total.append(df_split['len'].mean())

        #dir
        msin = np.mean(np.sin(df_split['dir_rad']))
        mcos = np.mean(np.cos(df_split['dir_rad']))
        avg_dir= np.arctan2(msin,mcos)
        all_avg_dir.append(avg_dir)

        #pol
        pol, pol_error = calc_polarization(df_split)
        all_pol.append(pol)
        all_pol_error.append(pol_error)

    assert(len(all_avg_len_total) == df_stats.shape[0])
    df_stats["meanBL_px_total"] = all_avg_len_total
    df_stats["meanDirection_rad_total"] = all_avg_dir
    df_stats["meanPolarization_total"] = all_pol
    df_stats["meanPolarizationError_total"] = all_pol_error
    
    return df_stats

def neighbor_calculations(df_allsplit, df_stats, json_files):
    # calculate neighbors for all datasets
    df_allsplit, data_numbers = calc_all_neighbors(df_allsplit, json_files)

    # calc class neighbors for all datasets
    df_allsplit = all_calc_class_neighbours(df_allsplit)

    # add same class neighbors percentage, average number of neighbors (per class and total), density, mean nnd and degree to stats
    df_stats = nb_stats_calculations(df_allsplit, df_stats)
    
    return df_allsplit, df_stats, data_numbers