#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 17:52:18 2018

@author: sudhir
"""
# =============================================================================
# Import packages
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# =============================================================================
# Read Geomertic function
# =============================================================================

def xyz(df1,df2,path,seed):

    def get_xyz_data(filename):
        pos_data = []
        lat_data = []
        with open(filename) as f:
            for line in f.readlines():
                x = line.split()
                if x[0] == 'atom':
                    pos_data.append([np.array(x[1:4], dtype=np.float),x[4]])
                elif x[0] == 'lattice_vector':
                    lat_data.append(np.array(x[1:4], dtype=np.float))
        return pos_data, np.array(lat_data)
        
    ga_cols =[]
    al_cols =[]
    o_cols = []
    in_cols = []
    
    for i in range(6):
        ga_cols.append('Ga'+str(i))
        al_cols.append('Al'+str(i))
        o_cols.append('O'+str(i))
        in_cols.append('In'+str(i))
    
    ga_df = pd.DataFrame(columns= ga_cols)
    al_df = pd.DataFrame(columns= al_cols)
    o_df = pd.DataFrame(columns= o_cols)
    in_df = pd.DataFrame(columns= in_cols)
        
    train = df1
    for i in train.id.values:
        fn = path+'train/{}/geometry.xyz'.format(i)
        train_xyz, train_lat = get_xyz_data(fn)
        
        ga_list = []
        al_list = []
        o_list = []
        in_list = []
        
        for li in train_xyz:
            try:
                if li[1] == "Ga":
                    ga_list.append(li[0])
            except:
                pass
            try:
                if li[1] == "Al":
                    al_list.append(li[0])
            except:
                pass
            try:
                if li[1] == "In":
                    in_list.append(li[0])
            except:
                pass
            try:
                if li[1] == "O":
                    o_list.append(li[0])
            except:
                pass
        
    #     ga_list = [item for sublist in ga_list for item in sublist]
    #     al_list = [item for sublist in al_list for item in sublist]
    #     o_list = [item for sublist in o_list for item in sublist]
       
        
        try:
            model = PCA(n_components=2,random_state=seed)
            ga_list = np.array(ga_list)
            temp_ga = model.fit_transform(ga_list.transpose())
            temp_ga = [item for sublist in temp_ga for item in sublist]
            
        except:
            temp_ga = [0,0,0,0,0,0]
    #         print i
        try:
            model = PCA(n_components=2 ,random_state=seed)
            al_list = np.array(al_list)
            temp_al = model.fit_transform(al_list.transpose())
            temp_al = [item for sublist in temp_al for item in sublist]
    #         print i
        except:
            temp_al = [0,0,0,0,0,0]
    #         print i
        try:
            model = PCA(n_components=2 ,random_state=seed)
            o_list = np.array(o_list)
            temp_o = model.fit_transform(o_list.transpose())
            temp_o = [item for sublist in temp_o for item in sublist]
    #         print i
        except:
            temp_o = [0,0,0,0,0,0]
    #         print i
        
        try:
            model = PCA(n_components=2 ,random_state=seed)
            in_list = np.array(in_list)
            temp_in = model.fit_transform(in_list.transpose())
            temp_in = [item for sublist in temp_in for item in sublist]
    #         print i
        except:
            temp_in = [0,0,0,0,0,0]
    #         print i
    
        temp_ga = pd.DataFrame(temp_ga).transpose()
        temp_ga.columns = ga_cols
        temp_ga.index = np.array([i])
    
        temp_al = pd.DataFrame(temp_al).transpose()
        temp_al.columns = al_cols
        temp_al.index = np.array([i])
    
        temp_o = pd.DataFrame(temp_o).transpose()
        temp_o.columns = o_cols
        temp_o.index = np.array([i])
        
        temp_in = pd.DataFrame(temp_in).transpose()
        temp_in.columns = in_cols
        temp_in.index = np.array([i])
        
        
    
        ga_df = pd.concat([ga_df,temp_ga])
        al_df = pd.concat([al_df,temp_al])
        o_df = pd.concat([o_df,temp_o])    
        in_df = pd.concat([in_df,temp_in])
        
    ga_df["id"] = ga_df.index
    al_df["id"] = al_df.index
    o_df["id"] = o_df.index
    in_df["id"] = in_df.index
    
    train = pd.merge(train,ga_df,on = ["id"],how = "left")
    train = pd.merge(train,al_df,on = ["id"],how = "left")
    train = pd.merge(train,o_df,on = ["id"],how = "left")
    train = pd.merge(train,in_df,on = ["id"],how = "left")
    
    # =============================================================================
    # Test data 
    # =============================================================================
    ga_df = pd.DataFrame(columns= ga_cols)
    al_df = pd.DataFrame(columns= al_cols)
    o_df = pd.DataFrame(columns= o_cols)
    in_df = pd.DataFrame(columns= in_cols)
        
    test = df2
    for i in test.id.values:
        fn = path+'test/{}/geometry.xyz'.format(i)
        train_xyz, train_lat = get_xyz_data(fn)
        
        ga_list = []
        al_list = []
        o_list = []
        in_list = []
        
        for li in train_xyz:
            try:
                if li[1] == "Ga":
                    ga_list.append(li[0])
            except:
                pass
            try:
                if li[1] == "Al":
                    al_list.append(li[0])
            except:
                pass
            try:
                if li[1] == "In":
                    in_list.append(li[0])
            except:
                pass
            try:
                if li[1] == "O":
                    o_list.append(li[0])
            except:
                pass
        
    #     ga_list = [item for sublist in ga_list for item in sublist]
    #     al_list = [item for sublist in al_list for item in sublist]
    #     o_list = [item for sublist in o_list for item in sublist]
       
        
        try:
            model = PCA(n_components=2 ,random_state=seed)
            ga_list = np.array(ga_list)
            temp_ga = model.fit_transform(ga_list.transpose())
            temp_ga = [item for sublist in temp_ga for item in sublist]
            
        except:
            temp_ga = [0,0,0,0,0,0]
    #         print i
        try:
            model = PCA(n_components=2 ,random_state=seed)
            al_list = np.array(al_list)
            temp_al = model.fit_transform(al_list.transpose())
            temp_al = [item for sublist in temp_al for item in sublist]
    #         print i
        except:
            temp_al = [0,0,0,0,0,0]
    #         print i
        try:
            model = PCA(n_components=2 ,random_state=seed)
            o_list = np.array(o_list)
            temp_o = model.fit_transform(o_list.transpose())
            temp_o = [item for sublist in temp_o for item in sublist]
    #         print i
        except:
            temp_o = [0,0,0,0,0,0]
    #         print i
        
        try:
            model = PCA(n_components=2 ,random_state=seed)
            in_list = np.array(in_list)
            temp_in = model.fit_transform(in_list.transpose())
            temp_in = [item for sublist in temp_in for item in sublist]
    #         print i
        except:
            temp_in = [0,0,0,0,0,0]
    #         print i
    
        temp_ga = pd.DataFrame(temp_ga).transpose()
        temp_ga.columns = ga_cols
        temp_ga.index = np.array([i])
    
        temp_al = pd.DataFrame(temp_al).transpose()
        temp_al.columns = al_cols
        temp_al.index = np.array([i])
    
        temp_o = pd.DataFrame(temp_o).transpose()
        temp_o.columns = o_cols
        temp_o.index = np.array([i])
        
        temp_in = pd.DataFrame(temp_in).transpose()
        temp_in.columns = in_cols
        temp_in.index = np.array([i])
        
        
    
        ga_df = pd.concat([ga_df,temp_ga])
        al_df = pd.concat([al_df,temp_al])
        o_df = pd.concat([o_df,temp_o])    
        in_df = pd.concat([in_df,temp_in])
        
    ga_df["id"] = ga_df.index
    al_df["id"] = al_df.index
    o_df["id"] = o_df.index
    in_df["id"] = in_df.index
    
    test = pd.merge(test,ga_df,on = ["id"],how = "left")
    test = pd.merge(test,al_df,on = ["id"],how = "left")
    test = pd.merge(test,o_df,on = ["id"],how = "left")
    test = pd.merge(test,in_df,on = ["id"],how = "left")
    
    train.In0 = train.In0.astype("float")
    train.In1 = train.In1.astype("float")
    train.In2 = train.In2.astype("float")
    train.In3 = train.In3.astype("float")
    train.In4 = train.In4.astype("float")
    train.In5 = train.In5.astype("float")
    
    test.In0 = test.In0.astype("float")
    test.In1 = test.In1.astype("float")
    test.In2 = test.In2.astype("float")
    test.In3 = test.In3.astype("float")
    test.In4 = test.In4.astype("float")
    test.In5 = test.In5.astype("float")
    
    
        
    return train,test 

