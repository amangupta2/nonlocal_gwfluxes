import math
import numpy as np
from netCDF4 import Dataset
from time import time as time2
import xarray as xr

import pandas as pd


f='/home/users/ag4680/myjupyter/137levs_ak_bk.npz'
data=np.load(f,mmap_mode='r')

lev=data['lev']
ht=data['ht']
ak=data['ak']
bk=data['bk']
R=287.05
T=250
rho = 100*lev/(R*T)


if 0:
	f = "/home/users/ag4680/myjupyter/era5_model_levels_table.xlsx"

if 0:
    data1 = pd.read_excel(f)

    era5_ml = data1['pf']
    lev     = era5_ml.to_numpy()
    ht      = data1['z_geom']/1000
    ak      = data1['a']
    bk      = data1['b']
    ht      = ht.to_numpy()
    ak      = ak.to_numpy()
    bk      = bk.to_numpy()
    ak = np.append([0],ak)
    bk = np.append([0],bk)

    R = 287.05
    T = 250
    rho = 100*lev/(R*T)

#outfile='/home/users/ag4680/myjupyter/137levs_ak_bk.npz'
#np.savez(outfile, lev=lev, ht=ht, ak=ak, bk=bk, R=R, T=T, rho=rho)
