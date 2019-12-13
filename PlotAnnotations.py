import pandas as pd
import numpy as np
from pylab import *
import os.path
import os
import json 
import glob

from tqdm import tqdm, tqdm_notebook
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#### Plot avg. positions on image
def plot_pos_cat_img(df_split, imgpath, number, show=True, save=False):

    #https://stackoverflow.com/questions/21654635/scatter-plots-in-pandas-pyplot-how-to-plot-by-category
    # show plot with corresponding image
    plt.close('all')

    fig, ax = plt.subplots()

    #datafile = cbook.get_sample_data('./data/IMG_5666.jpg')
    #img = imread(datafile)
    im = plt.imread(imgpath)
    implot = plt.imshow(im)

    groups = df_split.groupby('class')

    for name, group in groups:
        ax.plot(group.x_av, group.y_av, marker='o', linestyle='', ms=5, label=name)

    #plt.gca().invert_yaxis()
    plt.rcParams['figure.figsize'] = (20,10)
    
    if(save):
        if not os.path.exists('./output/plots'):
            os.makedirs('./output/plots')
        plt.savefig(os.path.join("./output/plots", str(number)+"_pos_img"))
    if(show):
        plt.show()
    
    print(f"Image {number}:  plotted positions ({imgpath})")
    
#### Plot avg. positions with orientation
#https://stackoverflow.com/questions/21654635/scatter-plots-in-pandas-pyplot-how-to-plot-by-category  
def plot_pos_ori_cat(df_split, imgpath, number, show=True, save=False):
    plt.close('all')

    fig, ax = plt.subplots()
    groups = df_split.groupby('class')

    for name, group in groups:
        #print(group.shape)
        ax.plot(group.xhead, group.yhead, marker='o', linestyle='', ms=10, label=name)

    df_plot = df_split.copy()[['xhead', 'ytail', 'yhead' , 'xtail']]
    for index, fish in df_plot.iterrows():
        ax.plot([fish[0],fish[3]], [fish[2],fish[1]], marker='None', linestyle='-', ms=1, color= 'Black')   

    ax.legend(loc = 'best')

    plt.gca().invert_yaxis()

    if(save):
        if not os.path.exists('./output/plots'):
            os.makedirs('./output/plots')
        plt.savefig(os.path.join("./output/plots/", str(number)+'_ori_cat'))
    if(show):
        plt.show()
    print(f"Image {number}:  plotted positions and orientations with categories ({imgpath})")
    
#### Plot avg. positions with orientation
#https://stackoverflow.com/questions/21654635/scatter-plots-in-pandas-pyplot-how-to-plot-by-category  
def plot_pos_ori_cat_img(df_split, imgpath, number, show=True, save=False):
    plt.close('all')

    fig, ax = plt.subplots()
    
    im = plt.imread(imgpath)
    implot = plt.imshow(im)
    
    groups = df_split.groupby('class')

    for name, group in groups:
        #print(group.shape)
        ax.plot(group.xhead, group.yhead, marker='o', linestyle='', ms=5, label=name)

    df_plot = df_split.copy()[['xhead', 'ytail', 'yhead' , 'xtail']]
    for index, fish in df_plot.iterrows():
        ax.plot([fish[0],fish[3]], [fish[2],fish[1]], marker='None', linestyle='-', ms=0.5, color= 'Black', alpha=0.5)   

    ax.legend(loc = 'best')
    
    plt.rcParams['figure.figsize'] = (20,10)

    #plt.gca().invert_yaxis()

    if(save):
        if not os.path.exists('./output/plots'):
            os.makedirs('./output/plots')
        plt.savefig(os.path.join("./output/plots/", str(number)+'_ori_cat_img'))
    if(show):
        plt.show()
    plt.close('all')

    
    print(f"Image {number}:  plotted positions and orientations with categories on image ({imgpath})")
    
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
    

def plot(df_allsplit, json_files, SHOW, SAVE):
    #get all numbers and imgpaths
    data_numbers = []
    img_paths = []

    for json in json_files:
        imgpath, number = get_imgpath_for_json(json)
        img_paths.append(imgpath)
        data_numbers.append(number)
        
    # generate plots for all datasets
    for i, dfsplit in enumerate(tqdm_notebook(df_allsplit)):
        plot_pos_cat_img(dfsplit, img_paths[i], data_numbers[i], show=SHOW, save=SAVE)
        plot_pos_ori_cat(dfsplit, img_paths[i], data_numbers[i], show=SHOW, save=SAVE)
        plot_pos_ori_cat_img(dfsplit, img_paths[i], data_numbers[i], show=SHOW, save=SAVE)
