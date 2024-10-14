import math
import numpy as np
#from matplotlib import pyplot as plt
from netCDF4 import Dataset
#import pygrib as pg
from time import time as time2
import xarray as xr
#import dask

#%matplotlib inline
import torch
import torch.nn as nn
import torch.nn.functional as F
# ------------ for data parallelism --------------------------
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
# ------------------------------------------------------------
import torch.optim as optim
#from torch.utils.data import DataLoader, random_split
#from torch.utils.data import Dataset as TensorDataset
from collections import OrderedDict
import pandas as pd


torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)

f='/home/users/ag4680/myjupyter/137levs_ak_bk.npz'
data=np.load(f,mmap_mode='r')

lev=data['lev']
ht=data['ht']
ak=data['ak']
bk=data['bk']
R=287.05
T=250
rho = 100*lev/(R*T)

# === Automatic Gradient Descent ===
from torch.nn.init import orthogonal_

def singular_value(p):
    sv = math.sqrt(p.shape[0]/p.shape[1])
    if p.dim() == 4:
        sv /= math.sqrt(p.shape[2]*p.shape[3])
    return sv

class AGD:
    @torch.no_grad()
    def __init__(self, model, gain=1.0):
        
        self.model   = model
        self.depth   = 0
        for p in model.parameters():
            if p.dim() == 2:
                self.depth += 1
        self.gain  = gain
        
        for p in model.parameters():
            #if p.dim() == 1:   
            if p.dim() == 2: 
                orthogonal_(p)
                p *= singular_value(p)
            if p.dim() == 4:
                for kx in range(p.shape[2]):
                    for ky in range(p.shape[3]):
                        orthogonal_(p[:,:,kx,ky])
                        p *= singular_value(p)

    
    @torch.no_grad()
    def step(self):
        
        G = 0
        for p in self.model.parameters():
            if p.dim() == 2:
                G += singular_value(p)*p.grad.norm(dim=(0,1)).sum()
        G /= self.depth
        
        log = math.log(0.5*(1 + math.sqrt( 1 + 4*G)))
        
        for p in self.model.parameters():
            if p.dim() == 2:
                factor = singular_value(p) / p.grad.norm(dim=(0,1), keepdim=True)
                p -= self.gain * log/self.depth * factor * p.grad


                # torch.cuda.is_available() checks and returns True if a GPU is available, else it'll return False

                
# https://stackoverflow.com/questions/54216920/how-to-use-multiple-gpus-in-pytorch
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # to select 1st GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # to select all available GPUs

#device="cpu"
#device = torch.device("cuda:1,3" if torch.cuda.is_available() else "cpu") # select the second and fourth GPU
#device="cpu"           

# 1. This is the best GPU check so far - If 4 GPUS, this should give an error for 4 and above, and only accept 0 to 3
# 2. https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
# 3. DistributedDataParallel is proven to be significantly faster than torch.nn.DataParallel for single-node multi-GPU data parallel training.

#torch.cuda.set_device(4) 

restart=False
init_epoch=1 # which epoch to resume from. Should have restart file from init_epoch-1 ready
nepochs=100

log_filename=f"./icml_train_ann-cnn_1x1_global_LR_4hl_hdim-2idim_restart_epoch_{init_epoch}_to_{init_epoch+nepochs-1}.txt"
#log_filename=f"./icml_train_ann-cnn_1x1_global_4hl_dropout0p1_hdim-2idim_restart_epoch_{init_epoch}_to_{init_epoch+nepochs-1}.txt"
def write_log(*args):
    line = ' '.join([str(a) for a in args])
    log_file = open(log_filename,"a")
    log_file.write(line+'\n')
    log_file.close()
    print(line)

if device != "cpu":
    ngpus=torch.cuda.device_count()
    print(f"NGPUS = {ngpus}")

write_log('In this Ablation study, learning rate is more variables between 1e-5 to 1e-3, rather than 1e-4 to 1e-3. 10 CPUs are requested and 8 CPUs are used.')

# redone files 2 - with constant scaling - larger collection
train_files = [
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2010_constant_mu_sigma_scaling01.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2010_constant_mu_sigma_scaling02.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2010_constant_mu_sigma_scaling03.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2010_constant_mu_sigma_scaling04.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2010_constant_mu_sigma_scaling05.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2010_constant_mu_sigma_scaling06.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2010_constant_mu_sigma_scaling07.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2010_constant_mu_sigma_scaling08.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2010_constant_mu_sigma_scaling09.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2010_constant_mu_sigma_scaling10.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2010_constant_mu_sigma_scaling11.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2010_constant_mu_sigma_scaling12.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2012_constant_mu_sigma_scaling01.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2012_constant_mu_sigma_scaling02.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2012_constant_mu_sigma_scaling03.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2012_constant_mu_sigma_scaling04.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2012_constant_mu_sigma_scaling05.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2012_constant_mu_sigma_scaling06.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2012_constant_mu_sigma_scaling07.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2012_constant_mu_sigma_scaling08.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2012_constant_mu_sigma_scaling09.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2012_constant_mu_sigma_scaling10.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2012_constant_mu_sigma_scaling11.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2012_constant_mu_sigma_scaling12.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2014_constant_mu_sigma_scaling01.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2014_constant_mu_sigma_scaling02.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2014_constant_mu_sigma_scaling03.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2014_constant_mu_sigma_scaling04.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2014_constant_mu_sigma_scaling05.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2014_constant_mu_sigma_scaling06.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2014_constant_mu_sigma_scaling07.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2014_constant_mu_sigma_scaling08.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2014_constant_mu_sigma_scaling09.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2014_constant_mu_sigma_scaling10.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2014_constant_mu_sigma_scaling11.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2014_constant_mu_sigma_scaling12.nc',
    
            ]       

test_files = [
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling01.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling02.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling03.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling04.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling06.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling07.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling08.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling09.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling10.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling11.nc',
    '/scratch/users/ag4680/training_data/era5/inputfeatures_u_v_theta_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling12.nc'
         ]


# final out of set testing on May 2015


# Customized Dataloader - good for loading batches from:
# -- carefully designed input files for both single-column and nonlocal global training, and 
# -- single point single-column and nonlocal training

# tricks to speed up pytorch dataloading: https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39
# dask scheduling suggestion: https://discuss.pytorch.org/t/problems-using-dataloader-for-dask-xarray-netcdf-data/108270
#dask.config.set(scheduler='synchronous')

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, files, domain, stencil, batch_size, manual_shuffle):

        #super().__init__()
        
        _files = sorted(files)
        
        # this concatenates all the files into a single index
        # chunk={"time":1} works the fastest
        ds: xr.Dataset = xr.open_mfdataset(
            paths=_files, chunks={"time": 1}, combine="nested", concat_dim="time"
        )
            
        self.ds = ds
        
        # dimensions
        self.idim = len(self.ds['idim'])
        self.odim = len(self.ds['odim'])
        
        
        self.inp = self.ds["features"]
        self.out = self.ds["output"]
        
        #self.lon = self.ds["lon"].to_numpy()
        #self.lat = self.ds["lat"].to_numpy()
        #self.nx = len(self.lon)
        #self.ny = len(self.lat)
        self.nt = len(self.ds.time)
        
        self.bs = batch_size
        self.domain=domain # acceptable values: singlepoint, regional, global
        self.stencil=stencil # for nonlocal training
        self.fac = int(self.stencil/2.)
        self.manual_shuffle=manual_shuffle
        #self.index = 0
        
        # create permutations
        if self.manual_shuffle:
            self.seed  = 51
            self.tperm = np.random.RandomState(seed=self.seed).permutation(np.arange(0,self.nt))
            
        if self.domain == 'singlepoint':
            self.y0 = 14
            self.x0 = 102
        
        if self.domain == 'regional':
            self.y1 = 10
            self.y2 = 18
            self.x1 = 98
            self.x2 = 106
            
    def __len__(self):
        if self.domain == 'singlepoint':
            return (self.nt)
        else:
            return (self.nt)
        
    def __getitem__(self, ind):
        
        if self.manual_shuffle:
            it = self.tperm[ind]
        else:
            it = ind
        
        if self.domain == 'singlepoint':
            # Note: assumes that the file is four dimensional (time, channels, lat, lon)
            if self.stencil == 1:
                I = torch.from_numpy(self.inp[it,:,self.y0,self.x0].data.compute())
                O = torch.from_numpy(self.out[it,:,self.y0,self.x0].data.compute())
                return I,O
            else:
                # Add boundary conditions!
                I = torch.from_numpy(self.inp[it,:,self.y0-self.fac:self.y0+self.fac+1,self.x0-self.fac:self.x0+self.fac+1].data.compute())
                O = torch.from_numpy(self.out[it,:,self.y0,self.x0].data.compute())
                return I,O 
        elif self.domain == 'regional':
            
            if self.stencil == 1:
                I = torch.from_numpy(self.inp[it,:,y1:y2,x1:x2].data.compute())
                O = torch.from_numpy(self.out[it,:,y1:y2,x1:x2].data.compute())
                
                #print(I.shape)
                #print(O.shape)
                # reshape
                I = torch.permute(I, (1,2,0))
                O = torch.permute(O, (1,2,0))
                S = I.shape
                I = I.reshape(S[0]*S[1], -1)
                O = O.reshape(S[0]*S[1], -1)
                
                return I,O 
            else:
                # (time x pressure x lat x lon x st x st)
                # convolution layer will be applied on the last two dimensions
                I = torch.from_numpy(self.inp[it,:,y1:y2,x1:x2,:,:].data.compute())
                O = torch.squeeze(torch.from_numpy(self.out[it,:,y1:y2,x1:x2,self.fac,self.fac].data.compute()))
            
        elif self.domain == 'global':
            if self.stencil == 1:
                I = torch.from_numpy(self.inp[it,:,:,:].data.compute())
                O = torch.from_numpy(self.out[it,:,:,:].data.compute())
                
                #print(I.shape)
                #print(O.shape)
                # reshape
                I = torch.permute(I, (1,2,0))
                O = torch.permute(O, (1,2,0))
                S = I.shape
                I = I.reshape(S[0]*S[1], -1)
                S = O.shape
                O = O.reshape(S[0]*S[1], -1)
                
                return I,O 
            else:
                # (time x pressure x lat x lon x st x st)
                # convolution layer will be applied on the last two dimensions
                I = torch.from_numpy(self.inp[it,:,:,:,:,:].data.compute())
                O = torch.squeeze(torch.from_numpy(self.out[it,:,:,:,self.fac,self.fac].data.compute()))
                # first reduce to a 4D tensor by vectorising to (time*lat*lon x pressure x st x st) shape
                I = torch.permute(I, (1,2,0,3,4))
                O = torch.permute(O, (1,2,0))
                S = I.shape
                I = I.reshape(S[0]*S[1], S[2], S[3], S[4])
                S = O.shape
                O = O.reshape(S[0]*S[1], -1)
                
                return I,O 
            
            
    def refresh_index(self):
        self.index = 1
        return True
    
    def return_ds(self):
        return self.ds

write_log('Done')

class ANN_CNN(nn.Module):
    
    def __init__(self,idim, odim, hdim, stencil, dropout=0):
        super().__init__()
        
        self.idim = idim
        self.odim = odim
        self.hdim = hdim
        self.dropout_prob = dropout
        self.stencil = stencil
        self.fac = np.floor(0.5*self.stencil)
        # assume normalized data as input
        # same activation for all layers
        # same dropout probabilities for all layers
        # ADD fac number of 3x3 CNN layers 
        # Either apply a 3x3, 5x5, 7x7 CNN layer, or apply multiple 3x3 layers
        if self.fac == 1:
            self.conv1 = nn.Conv2d(in_channels=idim, out_channels=idim, kernel_size=self.stencil, stride=1, padding=0)
            self.act_cnn = nn.ReLU()
            self.dropout0 = nn.Dropout(p=0.5*self.dropout_prob)
        elif self.fac == 2:
            self.conv1 = nn.Conv2d(in_channels=idim, out_channels=idim, kernel_size=self.stencil, stride=1, padding=0)
            self.act_cnn = nn.ReLU()
            self.dropout0 = nn.Dropout(p=0.5*self.dropout_prob)
        elif self.fac == 3:
            self.conv1 = nn.Conv2d(in_channels=idim, out_channels=idim, kernel_size=self.stencil, stride=1, padding=0)
            self.act_cnn = nn.ReLU()
            self.dropout0 = nn.Dropout(p=0.5*self.dropout_prob)
        
        self.layer1 = nn.Linear(idim,hdim)#,dtype=torch.float16)
        self.act1    = nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.GELU()#nn.ReLU()
        self.bnorm1   = nn.BatchNorm1d(hdim)
        self.dropout1 = nn.Dropout(p=0.5*self.dropout_prob)
        self.layer2 = nn.Linear(hdim,hdim)
        self.act2    = nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.GELU()#nn.ReLU()
        self.bnorm2   = nn.BatchNorm1d(hdim)
        self.dropout2 = nn.Dropout(p=self.dropout_prob)
        self.layer3 = nn.Linear(hdim,hdim)
        self.act3    = nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.GELU()#nn.ReLU()
        self.bnorm3   = nn.BatchNorm1d(hdim)
        self.dropout3 = nn.Dropout(p=self.dropout_prob)
        self.layer4 = nn.Linear(hdim,hdim)
        self.act4    = nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.GELU()#nn.ReLU()
        self.bnorm4   = nn.BatchNorm1d(2*hdim)
        self.dropout4 = nn.Dropout(p=self.dropout_prob)
        self.layer5 = nn.Linear(hdim,hdim)
        self.act5    = nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.GELU()#nn.ReLU()
        self.bnorm5   = nn.BatchNorm1d(hdim)
        self.dropout5 = nn.Dropout(p=self.dropout_prob)
        
        self.layer6 = nn.Linear(hdim,2*odim)
        self.act6    = nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.LeakyReLU()#nn.Tanh()#nn.GELU()#nn.ReLU()
        self.bnorm6   = nn.BatchNorm1d(2*odim)
        self.dropout6 = nn.Dropout(p=self.dropout_prob)
        
        self.output = nn.Linear(2*odim,odim) 
        
    def forward(self, x):
        
        if self.fac >= 1:
            x = torch.squeeze( self.dropout0(self.act_cnn(self.conv1(x))) )
            
        #elif fac == 2:
        #    x = self.conv1(x)
        #elif fac == 3:
            
        #print(f'new shape: {x.shape}')    
        x = self.dropout1(self.act1(self.layer1(x)))
        x = self.dropout2(self.act2(self.layer2(x)))
        x = self.dropout3(self.act3(self.layer3(x)))
        x = self.dropout4(self.act4(self.layer4(x)))
        x = self.dropout5(self.act5(self.layer5(x)))
        x = self.dropout6(self.act6(self.layer6(x)))
        x = self.output(x)
        
        #x = self.dropout1(self.bnorm1(self.act1(self.layer1(x))))
        #x = self.dropout2(self.bnorm2(self.act2(self.layer2(x))))
        #x = self.dropout3(self.bnorm3(self.act3(self.layer3(x))))
        #x = self.dropout4(self.bnorm4(self.act4(self.layer4(x))))
        #x = self.output(x)
        
        return x
    
    def gaussian_dropout(self, x):
        S = x.shape
        vec    = torch.normal( mean=torch.ones(S), std=torch.ones(S))
        return x*vec
    
    # calculates total number of learnable parameters
    def totalparams(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement()
            
        return param_size
    
    # computes total model size in MBs
    def totalsize(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        #print('model size: {:.3f}MB'.format(size_all_mb))
        
        return size_all_mb



def training(nepochs,model,optimizer,loss_fn,trainloader,testloader,stencil, bs_train,bs_test,save,
             file_prefix, init_epoch=1, scheduler=0):
    
    LOSS_TRAIN = np.zeros((nepochs))
    LOSS_TEST   = np.zeros((nepochs))
    
    for epoch in np.arange(init_epoch + 0, init_epoch + nepochs):
        # --------- training ----------
        model.train()
        trainloss=0.
        count=0.
        for i, (inp, out) in enumerate(trainloader):
            inp=inp.to(device)
            out=out.to(device)
            if stencil==1:
                S = inp.shape
                inp = inp.reshape(S[0]*S[1],S[2])
                S = out.shape
                out = out.reshape(S[0]*S[1],-1)
            elif stencil > 1:
                S = inp.shape
                inp = inp.reshape(S[0]*S[1],S[2],S[3], S[4])
                S = out.shape
                out = out.reshape(S[0]*S[1],-1)
            pred   =model(inp)
            loss     = loss_fn(pred,out)#loss_fn(pred*fac,out*fac) #+ weight_decay*l2_norm  #/fac) + 
            optimizer.zero_grad() # flush the gradients from the last step and set to zeros, they accumulate otherwise
            # backward propagation
            loss.backward()
            # parameter update step
            #print('5')
            optimizer.step()
            if scheduler !=0:
                scheduler.step()
            trainloss += loss#.item()#.item()
            count+=1
            
        LOSS_TRAIN[epoch-1-init_epoch] = trainloss/count
                
        # --------- testing ------------
        model.eval()
        #print('===== TESTING ============')
        testloss=0.
        count=0.
        for i, (inp, out) in enumerate(testloader):
            inp=inp.to(device)
            out=out.to(device)
            if stencil==1:
                S = inp.shape
                inp = inp.reshape(S[0]*S[1],S[2])
                S = out.shape
                out = out.reshape(S[0]*S[1],-1)
            elif stencil > 1:
                S = inp.shape
                inp = inp.reshape(S[0]*S[1],S[2],S[3], S[4])
                S = out.shape
                out = out.reshape(S[0]*S[1],-1)
            pred   =model(inp)
            loss2     = loss_fn(pred,out)
            testloss += loss2.item()
            count+=1
            
        LOSS_TEST[epoch-1-init_epoch] = testloss/count
        
        write_log(f'Epoch {epoch}, {(epoch-init_epoch+1)}/{nepochs}, training error: {LOSS_TRAIN[epoch-1-init_epoch]:.6f}, testing error: {LOSS_TEST[epoch-1-init_epoch]:.6f}')
        
        # Saving the model at any given epoch
        if save:
            savepath = f'/scratch/users/ag4680/{file_prefix}_train_epoch{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn,
                'loss_train': LOSS_TRAIN,
                'loss_test': LOSS_TEST,
                'activation': 'LeakyRelu()',
                'scheduler' : 'CyclicLR'
                }, savepath)
        
    return model, LOSS_TRAIN, LOSS_TEST#, EVOLVE




# setting Shuffle=True automatically takes care of permuting in time - but not control over seeding, so...
# set manual_shuffle=True and control seed from the function definition

# multiple batches from the vectorized matrices
bs_train=80#40#37
bs_test=80#40#37
# Can have multiple timesteps onto GPU per time step, 

# ============================= 1x1 ===========================
tstart=time2()

trainset1 = Dataset(files=train_files,domain='global', stencil=1, manual_shuffle=False, batch_size=bs_train)
trainloader1 = torch.utils.data.DataLoader(trainset1, batch_size=bs_train, 
                                          drop_last=False, shuffle=False, num_workers=8)#, persistent_workers=True)

testset1 = Dataset(files=test_files, domain='global', stencil=1, manual_shuffle=False, batch_size=bs_test)
testloader1 = torch.utils.data.DataLoader(testset1, batch_size=bs_test, 
                                         drop_last=False, shuffle=False, num_workers=8)#, persistent_workers=True)

tend=time2()
write_log(f'total_time={tend-tstart}')


# ================ 1x1 model ======================================================
idim    = trainset1.idim
odim    = trainset1.odim
hdim    = 2*idim

# lr 10-6 to 10-4 over 100 up and 100 down steps works well waise
model1     = ANN_CNN(idim=idim,odim=odim,hdim=hdim,dropout=0.2, stencil=trainset1.stencil)
model1 = model1.to(device)
write_log(f'model1 created. \n --- model1 size: {model1.totalsize():.2f} MBs,\n --- Num params: {model1.totalparams()/10**6:.3f} mil. ')
optim1     = optim.Adam(model1.parameters(),lr=1e-4)
scheduler1 = torch.optim.lr_scheduler.CyclicLR(optim1, base_lr=1e-5, max_lr=1e-3, step_size_up=50, step_size_down=50, cycle_momentum=False)



loss_fn    = nn.MSELoss()


fac = torch.ones(122)#torch.from_numpy(rho[15:]**0.1)
fac = (1./fac).to(torch.float32)
fac=fac.to(device)


print('Model created')



# new - with restart functionality
tstart=time2()

#restart=True

file_prefix = "torch_saved_models/icml_global/era5_global_ann_cnn_LR_leakyrelu_dropout0p1_cyclic_mseloss" 
#file_prefix = "torch_saved_models/icml_global/era5_global_ann_cnn_leakyrelu_NOdropout_cyclic_mseloss"
#nepochs=50 #30
if restart:
    #init_epoch = 9 # epoch to start new training from, reading from (epoch-1) file
    
    idim    = trainset1.idim
    odim    = trainset1.odim
    hdim    = 2*idim
    model1     = ANN_CNN(idim=idim,odim=odim,hdim=hdim,dropout=0.2, stencil=trainset1.stencil)
    write_log(f'model1 created. \n --- model1 size: {model1.totalsize():.2f} MBs,\n --- Num params: {model1.totalparams()/10**6:.3f} mil. ')
    model1 = model1.to(device) # important to make this transfer before the optimizer step in the next line. Otherwise eror
    optim1     = optim.Adam(model1.parameters(),lr=1e-4)
    scheduler1 = torch.optim.lr_scheduler.CyclicLR(optim1, base_lr=1e-4, max_lr=1e-3, step_size_up=50, step_size_down=50, cycle_momentum=False)
    PATH=f'/scratch/users/ag4680/{file_prefix}_train_epoch{init_epoch-1}.pt'
    checkpoint = torch.load(PATH)
    model1.load_state_dict(checkpoint['model_state_dict'])
    optim1.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch1 = checkpoint['epoch']
    loss_fn  = checkpoint['loss'] # same for all models
    
    #model1 = model1.to(device)
else:
    init_epoch = 1

write_log('Training...')
model1, loss_train1, loss_val1 = training(nepochs=nepochs, init_epoch=init_epoch,
                                model=model1, optimizer=optim1, loss_fn=loss_fn, 
                                trainloader=trainloader1, testloader=testloader1,
                                stencil=trainset1.stencil,
                                bs_train=bs_train, bs_test=bs_test,
                                save=True, 
                                file_prefix=file_prefix, 
                                scheduler=scheduler1
                               )
write_log('Done')
tend=time2()
print(f'total_time={tend-tstart}')