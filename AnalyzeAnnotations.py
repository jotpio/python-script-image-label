import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from pylab import *
import os.path
import os
import json 
import glob

from tqdm import tqdm, tqdm_notebook
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#### Load sloth jsons into array of pandas dataframes
def loadJSONintoDF(filepath):
    assert (os.path.isfile(filepath)) # check if file path exists
    annos = json.load( open(filepath, 'r') )[0]['annotations']
    df = pd.DataFrame(annos)
    return df
#### Load sloth jsons into array of pandas dataframes
def loadAllJSONSFromPath(datapath):
    assert (os.path.isdir(datapath)) # check if datapath is directory
    json_files = [pos_json for pos_json in os.listdir(datapath) if pos_json.endswith('.json')]
    
    #df_stats = pd.DataFrame(columns=['filename', '#fish', '#fish2', '#fish3'])
    all_df = []
    
    for filename in json_files:
        #df_stats = df_stats.append({'filename' : filename} , ignore_index=True)
        all_df.append(loadJSONintoDF(datapath + filename))
    df_stats = pd.DataFrame(0, index=json_files, columns=['#fish', '#fish2', '#fish3'])
    df_stats['#fish'] = 0
    df_stats['#fish2'] = 0
    df_stats['#fish3'] = 0
    df_stats['#allfishes'] = 0
    
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
        
        df_stats.at[json_files[i], '#fish'] = df_nclasses[0]
        df_stats.at[json_files[i], '#fish2'] = df_nclasses[1]
        df_stats.at[json_files[i], '#fish3'] = df_nclasses[2]
        df_stats.at[json_files[i], '#allfishes'] = sum(df_nclasses)
        
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

        df_split['pol_rad'] = np.arctan2((df_split['yhead']- df_split['ytail']), (df_split['xhead'] - df_split['xtail']))
        #print(df_split.head(1))

        df_allsplit.append(df_split)
    assert(len(df_allsplit) == len(json_files))
    
    return df_allsplit

def calc_neighbors(df_split, number):
    df_pos_av = df_split[['x_av', 'y_av']]

    dist_m = distance_matrix(df_pos_av, df_pos_av)
    df_dist_m = pd.DataFrame(dist_m)
    
    #find neighbours in distance matrix
    avg_length = df_split['len'].mean()
    dist1 = avg_length * 2;
    dist2 = avg_length * 4;

    print(f"{number}:   Distance1 (av2): {dist1:.5f} , Distance2 (av4): {dist2:.5f} ")

    #neighbors in distance average length*2
    df_dist_m_2 = df_dist_m[df_dist_m < dist1]
    df_dist_m_2 = df_dist_m_2[df_dist_m_2 != 0]
    assert(df_dist_m_2.shape[0] == df_split.shape[0])  #this should be the same as before

    np_dists = df_dist_m_2.values
    np_neighbor_pairs = np.argwhere(~np.isnan(np_dists))
    #np_neighbors2 = np.split(np_neighbor_pairs[:, 1], np.cumsum(np.unique(np_neighbor_pairs[:, 0], return_counts=True)[1])[:-1]) #https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function

    #setup neighbors
    np_neighbors2 = []
    for i_fish in range(df_dist_m_2.shape[0]):
        np_neighbors2.append([])
    #fill neighbors
    for neighbor_pair in np_neighbor_pairs:
        np_neighbors2[neighbor_pair[0]].append(neighbor_pair[1])
        
    assert(len(np_neighbors2) == df_split.shape[0])
    df_split['nb_av2'] = np_neighbors2

    #neighbors in distance average length*4
    df_dist_m_4 = df_dist_m[df_dist_m < dist2]
    df_dist_m_4 = df_dist_m_4[df_dist_m_4 != 0]
    assert(df_dist_m_4.shape[0] == df_split.shape[0])  #this should be the same as before

    np_dists = df_dist_m_4.values
    np_neighbor_pairs = np.argwhere(~np.isnan(np_dists))
    #np_neighbors4 = np.split(np_neighbor_pairs[:, 1], np.cumsum(np.unique(np_neighbor_pairs[:, 0], return_counts=True)[1])[:-1])    #https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function

    #setup neighbors
    np_neighbors4 = []
    for i_fish in range(df_dist_m_4.shape[0]):
        np_neighbors4.append([])
    #fill neighbors
    for neighbor_pair in np_neighbor_pairs:
        np_neighbors4[neighbor_pair[0]].append(neighbor_pair[1])
    
    assert(len(np_neighbors2) == df_split.shape[0])
    df_split['nb_av4'] = np_neighbors4
    
    return df_split
    
def calc_all_neighbors(df_allsplit, json_files):
    
    #get all numbers
    data_numbers = []
    for json in json_files:
        imgpath, number = get_imgpath_for_json(json)
        data_numbers.append(number)
        
    for i, df_split in enumerate(tqdm_notebook(df_allsplit)):
        calc_neighbors(df_split, data_numbers[i])
    return df_allsplit
        
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
    
    np_neighbors2 = df_split["nb_av2"].tolist()
    np_neighbors4 = df_split["nb_av4"].tolist()

    assert (len(np_neighbors2) == df_split.shape[0])
    assert (len(np_neighbors4) == df_split.shape[0])

    #count neighbors for dist1 = av * 2
    for nl in np_neighbors2:
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

    #count neighbors for dist2 = av * 4
    neighbor_list_av4_f = []
    neighbor_list_av4_f2 = []
    neighbor_list_av4_f3 = []

    for nl in np_neighbors4:
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

def all_calc_class_neighbours(df_allsplit):
    for df_split in tqdm_notebook(df_allsplit):
        calc_class_neighbors(df_split)
    return df_allsplit

def calc_same_class_nb_p_and_avg_number_nb(df_allsplit, df_stats):
    av2_allf_same = []
    av2_allf2_same = []
    av2_allf3_same = []

    av2_avg_total_n = []
    av2_avg_ff_n = []
    av2_avg_f2f2_n = []
    av2_avg_f3f3_n = []

    for i, df_split in enumerate(df_allsplit):

        av2_p_f_same = 0
        av2_p_f2_same = 0
        av2_p_f3_same = 0

        av2_avg_f_same = 0
        av2_avg_f2_same = 0
        av2_avg_f3_same = 0

        av2_num_f = df_stats["#fish"][i]
        av2_num_f2 = df_stats["#fish2"][i]
        av2_num_f3 = df_stats["#fish3"][i]

        av2_avg_n_len = 0

        # calculate average number of neighbors of same class of fish by all neighbors of fish per class and append to list of images
        # also calculate average number of same class neighbors and total neighbors
        for i in df_split.index:
            #avg number of same class neighbors and percentual
            c = df_split.at[i, "class"]
            av2_neighbors_len = len(df_split.at[i, "nb_av2"])

            av2_avg_n_len += av2_neighbors_len

            if(av2_neighbors_len == 0):
                continue

            if(c == "fish"):
                av2_p_f_same += df_split.at[i, "#neighbors_av2_f"] / av2_neighbors_len
                av2_avg_f_same += df_split.at[i, "#neighbors_av2_f"]
            elif(c == "fish_2"):
                av2_p_f2_same += df_split.at[i, "#neighbors_av2_f2"] / av2_neighbors_len
                av2_avg_f2_same += df_split.at[i, "#neighbors_av2_f2"]
            elif(c == "fish_3"):
                av2_p_f3_same += df_split.at[i, "#neighbors_av2_f3"] / av2_neighbors_len
                av2_avg_f3_same += df_split.at[i, "#neighbors_av2_f3"]

        av2_allf_same.append(av2_p_f_same/av2_num_f if av2_num_f != 0 else 0)
        av2_allf2_same.append(av2_p_f2_same/av2_num_f2 if av2_num_f2 != 0 else 0)
        av2_allf3_same.append(av2_p_f3_same/av2_num_f3 if av2_num_f3 != 0 else 0)

        av2_avg_total_n.append(av2_avg_n_len/df_split.shape[0] if df_split.shape[0] != 0 else 0)
        av2_avg_ff_n.append(av2_avg_f_same/av2_num_f if av2_num_f != 0 else 0)
        av2_avg_f2f2_n.append(av2_avg_f2_same/av2_num_f2 if av2_num_f2 != 0 else 0)
        av2_avg_f3f3_n.append(av2_avg_f3_same/av2_num_f3 if av2_num_f3 != 0 else 0)


    df_stats["p_av2_f_same"] = av2_allf_same
    df_stats["p_av2_f2_same"] = av2_allf2_same
    df_stats["p_av2_f3_same"] = av2_allf3_same

    df_stats["av2_avg_n_total"] = av2_avg_total_n
    df_stats["av2_avg_n_f"] = av2_avg_ff_n
    df_stats["av2_avg_n_f2"] = av2_avg_f2f2_n
    df_stats["av2_avg_n_f3"] = av2_avg_f3f3_n
    
    # distance: av4
    
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

        av4_num_f = df_stats["#fish"][i]
        av4_num_f2 = df_stats["#fish2"][i]
        av4_num_f3 = df_stats["#fish3"][i]

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


    df_stats["p_av4_f_same"] = av4_allf_same
    df_stats["p_av4_f2_same"] = av4_allf2_same
    df_stats["p_av4_f3_same"] = av4_allf3_same

    df_stats["av4_avg_n_total"] = av4_avg_total_n
    df_stats["av4_avg_n_f"] = av4_avg_ff_n
    df_stats["av4_avg_n_f2"] = av4_avg_f2f2_n
    df_stats["av4_avg_n_f3"] = av4_avg_f3f3_n

    return df_stats