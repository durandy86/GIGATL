#!/usr/bin/python3 python
# coding: utf-8

# In[1]:


import sys
import os
import numpy as np
import xarray as xr
import pandas as pd
from xgcm import Grid
import dask.array as da
import pyinterp

import gc

from definition.defPrincipal5 import *
from definition.computeWavelet import *

# import multiprocessing as mp

# In[2]:


from dask.distributed import Client, LocalCluster
#
# Initialisation d'un cluster de 32 coeurs
cluster = LocalCluster(processes=False, n_workers=1, threads_per_worker=48)
client = Client(cluster)
client


# In[3]:


# path='/media/durand/Gigatl/data_avril2010/'
path = '/home/datawork-lops-megatl/GIGATL6/GIGATL6_1h/HIS/'
#V = ds.data_vars
filenames = path + pd.read_csv('liste',header=None)
filenames = filenames.values.flatten().tolist()


# ds = xr.open_dataset(filenames[0], chunks={'time_counter': 1,'s_rho': 1},
#                      drop_variables=['time', 'sustr', 'svstr','salt','temp','w','bvf',
#                                      'hc','h','Vtransform','sc_r','sc_w','Cs_r','Cs_w','angle','mask_rho',
#                                       'pm','pn','Tcline','theta_s','theta_b','f',
#                                       'lon_rho','lat_rho','lon_u','lat_u','lon_v','lat_v'])
# datasets.append(ds)


# In[ ]:





# In[4]:


ds = xr.open_dataset(filenames[0], chunks={'time_counter': 1,'s_rho': 1},
                     drop_variables=['time', 'sustr', 'svstr','salt','temp','w','bvf',
                                     'hc','h','Vtransform','sc_r','sc_w','Cs_r','Cs_w','angle','mask_rho',
                                      'pm','pn','Tcline','theta_s','theta_b','f',
                                      'lon_rho','lat_rho','lon_u','lat_u','lon_v','lat_v'])

# On récupère la liste des fichiers à ouvrir via le fichier liste 
path = "/home/datawork-lops-megatl/GIGATL6/GIGATL6_1h/HIS/"
gridname = path+'GIGATL6_12h_inst_2004-01-15-2004-01-19.nc'
gd = xr.open_dataset(gridname, chunks={'s_rho': 1})
ds['hc'] = gd.hc
ds['h'] = gd.h
ds['Vtransform'] = gd.Vtransform
ds['sc_r'] = gd.sc_r
ds['sc_w'] = gd.sc_w
ds['Cs_r'] = gd.Cs_r
ds['Cs_w'] = gd.Cs_w
ds['angle'] = gd.angle
ds['mask_rho'] = gd.mask_rho

# On modifie des dimensions et des coordonnées, on crée la grille xgcm
ds = adjust_grid(ds)
L = ds.dims['x_rho']
M = ds.dims['y_rho']
N = ds.dims['s_rho']

# On crée la grille xgcm
ds = xgcm_grid(ds)
grid = ds.attrs['xgcm-Grid']


# In[5]:


ds2 = ds.isel(time_counter=0)

ds2 = ds2.isel(x_rho=slice(300,None), x_u=slice(301,None),
             y_rho=slice(360,1300), y_v=slice(3611,1300)).load()

z_r = get_z(ds2,zeta=ds2['zeta'],hgrid='r')


# In[ ]:


dt = ( 12.* 3600.)
# keep one step per day
step = int(86400. / dt)
# ds = ds.isel(time_counter=slice(0,None,step))


# Initialize new cartesian grid for interpolation
nx, ny = (5370, 2000)
x = np.linspace(-65., 15., nx)
y = np.linspace(-20., 20., ny)

depth_Ana=[-1500,-2000.]
for d in depth_Ana : 
    month=1
    nDay=0
    datasets = []
    for f in filenames :
        ds = xr.open_dataset(f, chunks={'time_counter': 1,'s_rho': 1},
                             drop_variables=['time', 'sustr', 'svstr','salt','temp','w','bvf',
                                              'hc','h','Vtransform','sc_r','sc_w','Cs_r','Cs_w','angle','mask_rho',
                                              'pm','pn','Tcline','theta_s','theta_b','f',
                                              'lon_rho','lat_rho','lon_u','lat_u','lon_v','lat_v'
                                              'time_instant','time_instant_bounds'])

        ds=ds.isel(time_counter=slice(0,None,step))
        
        ds['hc'] = gd.hc
        ds['h'] = gd.h
        ds['Vtransform'] = gd.Vtransform
        ds['sc_r'] = gd.sc_r
        ds['sc_w'] = gd.sc_w
        ds['Cs_r'] = gd.Cs_r
        ds['Cs_w'] = gd.Cs_w
        ds['angle'] = gd.angle
        ds['mask_rho'] = gd.mask_rho

        # On modifie des dimensions et des coordonnées, on crée la grille xgcm
        ds = adjust_grid(ds)
        L = ds.dims['x_rho']
        M = ds.dims['y_rho']
        N = ds.dims['s_rho']

        # On crée la grille xgcm
        ds = xgcm_grid(ds)
        grid = ds.attrs['xgcm-Grid']
        
        for i in range(0,len(ds.time_counter)):
        # extrait une partie de ds temporelle et spatiale horizontalement

            ds2 = ds.isel(time_counter=i)
            [urot,vrot] = rotuv(ds2,'r')
            ds2['vrot'] = vrot

            ds2 = ds2.isel(x_rho=slice(300,None), x_u=slice(301,None),
                         y_rho=slice(360,1300), y_v=slice(3611,1300)).load()
            
            # Initialise grid at specific depth
            depth = d
            vslice = slice2(ds2, ds2.vrot, z_r, depth=depth)

            # Compute interpolation on new grid
            v_cart = rtree_xr(ds2, vslice, x, y)
            
            v_cart = v_cart.fillna(0)
            
            #Conconate v_cart
            v_cart = v_cart.assign_coords(time_counter=ds.time_counter[i].astype('datetime64'))
            v_cart.time_counter.encoding['units'] = "seconds since 1979-01-01 00:00:00"
            v_cart.time_counter.encoding['calendar'] = "gregorian"
            datasets.append(v_cart)
            
            nDay=nDay+1
            print(nDay)

        ds3 = xr.concat(datasets, dim='time_counter', coords='minimal', compat='override')
        ds3 = ds3.to_dataset(name='vCart')
        ds3 = ds3.rename_dims({'x_rho':'x','y_rho':'y'})
        
        # del z_r,vslice,v_cart,ds,ds2,datasets,gd
        # gc.collect()

        xxx=(ds3.coords['x_rho']).astype('float32')
        yyy=(ds3.coords['y_rho']).astype('float32')

        ds3.coords['x_rho']=xxx
        ds3.coords['y_rho']=yyy

        yLevelMin=find_nearest(ds3.y_rho, -2.)
        yLevelMax=find_nearest(ds3.y_rho, 2.)
        ds3=ds3.isel(y=slice(yLevelMin,yLevelMax))        
        
        if nDay%30 == 0 : 
            del ds3['time_instant']
            print(ds3)
            ds3.to_netcdf('exp_2011_equator_d'+str(int(abs(d)))+'_m0'+str(month)+'.nc')

            nDay=0
            month=month+1
            datasets = []
            gc.collect()


# In[ ]:


ds3.time_counter.values


# In[ ]:


cluster.close()

