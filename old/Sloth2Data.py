import json 
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix


def SummaryByXXX(df, xxx=None):
    if xxx is None:
        xxx = 'class'
    classes = np.unique(df[xxx].values)
    dfout = pd.DataFrame()
    dfout[xxx] = pd.Series(classes)
    dfout['abundance'] = pd.Series( np.zeros(len(classes)) )
    for i, cla in enumerate(classes):
        dfout.loc[dfout[xxx] == cla, 'abundance'] = len( df.loc[df[xxx] == cla] )
    return dfout


def SplitPosition(df):
    '''
    sloth annotation has xn and yn with
    xn = xhead;xtail
    yn = yhead;ytail
    this fctn. splits the yn and xn to new columns in the dataframe
    '''
    xn = np.array([ np.array(xn.split(';'), dtype=float) for xn in df['xn']])
    dfout['xhead'] = pd.Series( xn[:, 0] )
    dfout['xtail'] = pd.Series( xn[:, 1] )
    yn = np.array([ np.array(yn.split(';'), dtype=float) for yn in df['yn']])
    dfout['yhead'] = pd.Series( yn[:, 0] )
    dfout['ytail'] = pd.Series( yn[:, 1] )


def main():
    f_name = './CLIP0000306_000_FT0400_f0060.json'
    annos = json.load( open(f_name, 'r') )[0]['annotations']
    df = pd.DataFrame(annos)
    # print(type(df['xn'].values[0]))
    # print([ xn for xn in df['xn']])
    print(np.array(df['xn'][0].split(';'), dtype=float))
    SplitPosition(df)
    print(df.head())
    dfSum = SummaryByXXX(df, xxx='class')
    print( dfSum.head() )

    
if __name__ == '__main__':
    main()
