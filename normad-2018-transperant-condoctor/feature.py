#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 07:50:54 2018

@author: sudhir
"""

import numpy as np
import pandas as pd

def feature_engineering(data):
    data = data.copy()
    # Degree to radian
    def degree_radians(df):
        print('# Degree to radian')
        df['alpha_rad'] = np.radians(df['lattice_angle_alpha_degree'])
        df['beta_rad'] = np.radians(df['lattice_angle_beta_degree'])
        df['gamma_rad'] = np.radians(df['lattice_angle_gamma_degree'])
    
    # Trinomerty
    """def sin_cos_tan(df):
        print('# Sin cos tan')
        col = ['alpha_rad','beta_rad','gamma_rad']
        for c in col:
            df[c+'_sin_theta'] = np.sin(df[c])
            df[c+'_cos_theta'] = np.cos(df[c])
            df[c+'_tan_theta'] = np.tan(df[c])"""
    
    #Volumn
    def vol(df):
        """ Args:
            a (float) - lattice vector 1
            b (float) - lattice vector 2
            c (float) - lattice vector 3
            alpha (float) - lattice angle 1 [radians]
            beta (float) - lattice angle 2 [radians]
            gamma (float) - lattice angle 3 [radians]
            Returns:
            volume (float) of the parallelepiped unit cell"""
        print('# Volumn')    
        volumn = df['lattice_vector_1_ang']*df['lattice_vector_2_ang']*df['lattice_vector_3_ang']*np.sqrt(
        1 + 2*np.cos(df['alpha_rad'])*np.cos(df['beta_rad'])*np.cos(df['gamma_rad'])
        -np.cos(df['alpha_rad'])**2
        -np.cos(df['beta_rad'])**2
        -np.cos(df['gamma_rad'])**2)
        df['volumn'] = volumn
        
    
    # Atomic density
    def atomic_density(df):
        print('# Atomic density')
        df['density'] = df['number_of_total_atoms'] / df['volumn']
    
    # Mean & Median range
    def mean_median_feature(df):
        print('# Mean & Median range')
        dmean = df.mean()
        dmedian = df.median()
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        col = df.columns
        del_col = ['id','formation_energy_ev_natom','bandgap_energy_ev']
        col = [w for w in col if w not in del_col]
        
        for c in col:
            df['mean_'+c] = (df[c] > dmean[c]).astype(np.uint8)
            df['median_'+c] = (df[c] > dmedian[c]).astype(np.uint8)
            df['q1_'+c] = (df[c] < q1[c]).astype(np.uint8)
            df['q3_'+c] = (df[c] > q3[c]).astype(np.uint8)
        print('Shape',df.shape)
    
    degree_radians(data)
    #sin_cos_tan(data)
    vol(data)
    atomic_density(data)
    mean_median_feature(data)
    return data
 

# One Hot Encoding
def OHE(df1,df2,columns):
    print('# One Hot Encoding')
    len = df1.shape[0]
    df = pd.concat([df1,df2],axis=0)
    c2,c3 = [], {}
    print('Categorical variables',columns)
    for c in columns:
        c2.append(c)
        c3[c] = 'ohe_'+c
        
    df = pd.get_dummies(data = df, columns = c2, prefix = c3)
    df1 = df.iloc[:len,:]
    df2 = df.iloc[len:,:]
    print('Data size',df1.shape,df2.shape)
    return df1,df2

    
    