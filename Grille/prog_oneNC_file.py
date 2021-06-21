#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import numpy as np
import xarray as xr
import pandas as pd
import dask.array as da


# In[2]:


from dask.distributed import Client, LocalCluster
#
# Initialisation d'un cluster de 32 coeurs
cluster = LocalCluster(processes=False, n_workers=1, threads_per_worker=32)
client = Client(cluster)
client


# In[5]:


path = './'
filenames = path + pd.read_csv('listeForWavelet',header=None)
filenames = filenames.values.flatten().tolist()

datasets=[]
for f in filenames:
    ds = xr.open_dataset(f,chunks={'time_counter': 1})
    datasets.append(ds)
ds = xr.concat(datasets, dim='time_counter', coords='minimal', compat='override')


# In[6]:


ds.to_netcdf('exp_2011_equator_1000m.nc')


# In[7]:


cluster.close()


# In[ ]:




