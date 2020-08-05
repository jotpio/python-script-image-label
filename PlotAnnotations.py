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
def plot_pos_cat_img(df_split, imgpath, show=True, save=False):
    number = df_split.number
    
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
def plot_pos_ori_cat(df_split, show=True, save=False):
    number = df_split.number
    
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
    print(f"Image {number}:  plotted positions and orientations with categories")
    
#### Plot avg. positions with orientation
#https://stackoverflow.com/questions/21654635/scatter-plots-in-pandas-pyplot-how-to-plot-by-category  
def plot_pos_ori_cat_img(df_split, imgpath, show=True, save=False):
    number = df_split.number
    
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
# def get_imgpath_for_json(filename, all_imgdir):
#     #print(filename)
#     number = filename[4:8]
        
#     imgfiles = [img_name for img_name in os.listdir(all_imgdir) if img_name.endswith('.jpg')]
#     found_file = -1
    
#     for name in imgfiles:
#         if name[4: -4] == number:
#             found_file = name
#     assert(found_file != -1)
#     imgpath = all_imgdir + found_file
#     assert (os.path.isfile(imgpath))
    
#     return imgpath, number
def get_imgpath_for_number(number, all_imgdir):
        
    imgfiles = [img_name for img_name in os.listdir(all_imgdir) if (img_name.endswith('.jpg') or img_name.endswith('.JPG'))]
    found_file = -1
    
    for imgname in imgfiles:
        if imgname[-8: -4] == number:
            found_file = imgname
#     assert(found_file != -1), f"No image found with number {number} in last 4 digits (...XXXX.jpg)"
    if(found_file != -1):
        imgpath = all_imgdir + found_file
        assert (os.path.isfile(imgpath))
        return imgpath
    else:
        print(f"      No image found for dataset with number: {number}")
        return -1
    

def plot(df_allsplit, json_files, all_imgdir, SHOW, SAVE):
    #get imgpaths
#     img_paths = []

#     for json in json_files:
#         imgpath = get_imgpath_for_json(json, all_imgdir)
#         img_paths.append(imgpath)
        
    # generate plots for all datasets
    for i, df_split in enumerate(tqdm_notebook(df_allsplit)):
        img_path = get_imgpath_for_number(df_split.number, all_imgdir)
        if (img_path != -1):
            plot_pos_cat_img(df_split, img_path, show=SHOW, save=SAVE)
            plot_pos_ori_cat_img(df_split, img_path, show=SHOW, save=SAVE)
        plot_pos_ori_cat(df_split, show=SHOW, save=SAVE)

            
