#!/usr/bin/env python
# coding: utf-8


import numpy as np
import xarray as xr
import pandas as pd
import dask.array as da

import time

import pywt

from definition.basic_func import *

deg2m = 111000


def compute_wavelet_mode(indir,exp,var,FourierDim,**kwargs):

    """
    Compute for a given vertical mode (bar or 1, 2, 3 ...etc)
    the wavelet transform along the FourierDim ('time', 'x' or 'y') 
    of the variable var ('u' or 'v') for the experiment exp.
    
    ** options
        - time_avg: false: consider the full time series of ti 
                    true : averga over ti before performing wt.

    :return: netcdf file containing the wavelet power spectrum
    as a function of (time,FourierVar,y,x).
    """

    options={
            'time_avg':False
            }
    
    options.update(kwargs)

    print('Compute '+FourierDim+'-wavelet for exp: '+exp)

    data = load_modes(exp)
    
    if options['time_avg']==True :
        # average over ti
        mat = data[var+mode].mean('time_counter')
        mat = mat.expand_dims({'time_counter':np.array(['avg'])},axis=0)
    else:
        mat = data[var]
    
    # Variable extraction
    x = mat['x'].values
    y = mat['y'].values
    lat = mat['y_rho'][:].values
    lon = mat['x_rho'][:].values
    time = mat.time_counter.values
    
    # Resolution
    if options['time_avg']==False:
        dt = np.diff(time).mean().astype('timedelta64[s]').astype('float64') # default: dt=10*86400

    dx = np.diff(lon).mean()*deg2m # default: dx=0.25*111000
    dy = np.diff(lat).mean()*deg2m # default: dy=0.25*111000
    
    # define variable name for output netcdf
    namevarf = var+'hat'

    if FourierDim == 'time':
    
        wavelet_type = "cmor1-1.5"
        scale_freqcy = np.arange(1,50,0.5)

        fourier_scale = 1/(pywt.scale2frequency(wavelet_type,scale_freqcy)/dt)

        mhat = comp_wavelet_mat_omega(mat,x,y,time,dt,wavelet_type,scale_freqcy)
    
        # write to netcdf format
        vhat=xr.Dataset({namevarf:(['time_counter','periods','y','x'],mhat)},coords={'time_counter':time,'periods':fourier_scale,'y':y,'x':x,'lat':lat,'lon':lon})
        print(vhat)
        
        
    if FourierDim == 'x':
    
        wavelet_type = "cmor1-1.5"
        scale_freqcy = np.arange(5,200,1)

        fourier_scale = 1/(pywt.scale2frequency(wavelet_type,scale_freqcy)/dx)

        mhat = comp_wavelet_mat_kx(mat,x,y,lon,lat,time,dx,wavelet_type,scale_freqcy)
    
        # write to netcdf format
        vhat=xr.Dataset({namevarf:(['time_counter','xWavelength','y','x'],mhat)},coords={'time_counter':time,'xWavelength':fourier_scale,'y':y,'x':x,'lat':lat,'lon':lon})

    if FourierDim == 'y':
    
        wavelet_type = "cmor1-1.5"
        scale_freqcy = np.arange(5,200,1)

        fourier_scale = 1/(pywt.scale2frequency(wavelet_type,scale_freqcy)/dy)

        mhat = comp_wavelet_mat_ky(mat,x,y,lon,lat,time,dx,wavelet_type,scale_freqcy)
    
        # write to netcdf format
        vhat=xr.Dataset({namevarf:(['time_counter','yWavelength','y','x'],mhat)},coords={'time_counter':time,'yWavelength':fourier_scale,'y':y,'x':x,'lat':lat,'lon':lon})

    # write to netcdf file
    if options['time_avg']==False:   
        vhat.to_netcdf(indir+'Eq_'+var+'hat_'+'exp_'+FourierDim+'Fourier.nc') 
    else:
        vhat.to_netcdf(indir+exp+'/Eq_'+var+'hat_'+mode+'_'+FourierDim+'Fourier_avg.nc') 
    
    print('DONE : Compute '+FourierDim+'-wavelet for exp: '+exp)

    return 


def comp_wavelet_mat_kx(mat,x,y,lon,lat,time,dx,wavelet_type,scale_freqcy):
    
    """
    Compute zonal wavelet transform for a matrix with dimensions (t,y,x)
    typically used for the velocity projected on the different vertical modes
    Ubar(t,y,x) or Ucline1(t,y,x), Ucline2(t,y,x) ...

    :return: matrix mhat with dimension (t, x-WAVELENGTH, y, x) and unit [mat]**2
    """
    xWavelength = 1/(pywt.scale2frequency(wavelet_type,scale_freqcy)/dx)
    
    lon0 = lon[0]
    lonN = lon[-1]
    
    mhat = np.nan*np.zeros((time.size,xWavelength.size,y.size,x.size),dtype='float32')

    for t in range(len(time)):
        for j in range(len(y)):

            vect = mat[t,j,:].values

            # apply wavelet transform
            amp, _  = pywt.cwt(vect,scale_freqcy,wavelet_type,sampling_period=dx)
            
            # convert complex amplitude to real ampitude
            mhat[t,:,j,:] = (abs(amp)**2)

            # remove area outside the cone of influence            
            for l in range(len(xWavelength)):
                mhat[t,l,j,:][lon<lon0+xWavelength[l]/2/deg2m] = np.nan
                mhat[t,l,j,:][lon>lonN-xWavelength[l]/2/deg2m] = np.nan

    return mhat 


def wtMax2nc(indir,exp,varhat,FourierDim,**kwargs):

    """
    Extract the values of the fourier dimension FourierDim ('time', 'x' or 'y')
    for the maximum wavelet power spectrum of variable varhat ('uhat' or 'vhat')
    projected on the vertical mode mode ('bar', 'cline1', 'cline2' ...etc) for 
    the experiment exp over the period ti.
    ! parallel computing 
    
    **options:
        - time_avg: True or False: consider the averaged 
                    or full time series wavelet transform.

    call: compute_wtFourierMax_mode

    :return: netcdf file with 
    """

    options={
       'time_avg':False
    }

    options.update(kwargs)

    if options['time_avg']==False:
          data = xr.open_dataset(indir+exp)
#         data = xr.open_dataset(indir+exp+'/Eq_'+varhat+'_'+mode+'_'+FourierDim+'Fourier_'+ti+'.nc')
    else:
          print('coucou')
#         data = xr.open_dataset(indir+exp+'/Eq_'+varhat+'_'+mode+'_'+FourierDim+'Fourier_'+ti+'_avg.nc')
    
    # Extract variables

    varh=data[varhat].values
    
    if (FourierDim=='x') :
        FourierDim_vect=data['xWavelength'].values
        namevar='KXmax'
    if (FourierDim=='y') :
        FourierDim_vect=data['yWavelength'].values
        namevar='KYmax'
    if FourierDim=='time':
        FourierDim_vect=data['periods'].values
        namevar='Pmax'

    x = data.x.values
    y = data.y.values
    time = data.time_counter.values
    lat = data.lat.values
    lon = data.lon.values

#     # Preprocess parallel computing: split
    
#     nblocks, dims = (time.size),(0)

#     varhat_spl = splitter1D(varh, nblocks, dims)

#     arg_dict = [{'varhat':varhat_spl[i],'FourierDim_vect':FourierDim_vect} for i in range(len(varhat_spl))]

#     # Parallel computing
#     pool = mp.Pool(64,maxtasksperchild=1)  # use 64 proc max on santo machine
#     output = pool.map(compute_wtFourierMax_mode_wrapper,arg_dict,chunksize=1)

#     pool.close()
#     pool.join()

    # Postprocess parallel computing: stitch
#     output_zipped = list(zip(*output))
    
#     Fmax = stitcher1D(output_zipped[0], dims)
#     Famp = stitcher1D(output_zipped[1], dims)
    
    Fmax, Famp = compute_wtFourierMax_mode(varh,FourierDim_vect)
    # write to netcdf
    kmax = xr.Dataset({namevar:(['time_counter','y','x'],Fmax),'amp':(['time_counter','y','x'],Famp)},coords={'time_counter':time,'y':y,'x':x,'lat':lat,'lon':lon})

    if options['time_avg']==False:
        kmax.to_netcdf(indir+'/Eq_'+namevar+'_'+varhat+'2.nc')
    else:
        kmax.to_netcdf(indir+exp+'/Eq_'+namevar+'_'+varhat+'_'+mode+'_'+ti+'_avg.nc')
    
    print('Done')
    return

def compute_wtFourierMax_mode(varhat,FourierDim_vect):

    """
    Extract the dominant values of the FourierDim from the 
    3d (Dim, FourierDim, Amp) wavelet transform output 
    FourierDim_vect : vector of period or wavelength

    :return: Fmax the table of the value of the fourier variable
    of the maximum power spectrum amplitude at each grid cell (t,i,j) 
    and Famp the amplitude of the maximum.
    """

    time_size = varhat.shape[0]
    y_size = varhat.shape[2]
    x_size = varhat.shape[3]

    Fmax = np.nan*np.zeros((time_size,y_size,x_size))
    Famp = np.nan*np.zeros((time_size,y_size,x_size))
    
    for t in range(time_size):
        for i in range(x_size):
            for j in range(y_size):

                try:
                    amp_max = my_nanmax(varhat[t,:,j,i],0)
                    ind_max = my_argmax(varhat[t,:,j,i],0)

                    Fmax[t,j,i] = FourierDim_vect[ind_max] 
                    Famp[t,j,i] = amp_max

                except:
                    # all nan values.
                    pass


    return Fmax, Famp


def compute_wavelet_mode_wrapper(arg_dict):

    return compute_wavelet_mode(**arg_dict)