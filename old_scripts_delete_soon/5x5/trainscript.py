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

                
# https://stackoverflow.com/questions/54216920/how-to-use-multiple-gpus-in-pytorch
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # to select 1st GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # to select all available GPUs

# 1. This is the best GPU check so far - If 4 GPUS, this should give an error for 4 and above, and only accept 0 to 3
# 2. https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
# 3. DistributedDataParallel is proven to be significantly faster than torch.nn.DataParallel for single-node multi-GPU data parallel training.
#torch.cuda.set_device(4) 

# IMPORTANT RUN PARAMETERS TO SET BEFORE SUBMISSION
restart=False
init_epoch=1 # which epoch to resume from. Should have restart file from init_epoch-1 ready
nepochs=100

stencil=5
domain='global' # or 'stratosphere_only'


if domain == 'global':
    log_filename=f"./global_{stencil}x{stencil}_uvthetaw_6hl_epoch_{init_epoch}_to_{init_epoch+nepochs-1}.txt"
elif domain == 'stratosphere_only':
    log_filename=f"./ss_only_{stencil}x{stencil}_uvthetaw_6hl_epoch_{init_epoch}_to_{init_epoch+nepochs-1}.txt"
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

write_log(f'Retraining the {stencil}x{stencil} global model with u,v,theta,w after ICML. Revisiting and checking why validation error was so high with the u,v,theta runs. Trainset = 2010+2012+2014 + all except may 2015. Validation set = May 2015.')

# ====================================================================================================
# DEFINING INPUT FILES
if stencil == 1:
	pref=f'{stencil}x{stencil}'
elif stencil > 1:
        pref=f'nonlocal_{stencil}x{stencil}'

if domain == 'global':
    pre='/scratch/users/ag4680/training_data/era5/'+pref+'_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_'
elif domain == 'stratosphere_only':
    pre='/scratch/users/ag4680/training_data/era5/stratosphere_'+pref+'_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_'
write_log(f'File prefix: {pre}')

# Deleting March 2015 from training data due to corrupted data
# redone files 2 - with constant scaling - larger collection
train_files = [
    pre+'2010_constant_mu_sigma_scaling01.nc',
    pre+'2010_constant_mu_sigma_scaling02.nc',
    pre+'2010_constant_mu_sigma_scaling03.nc',
    pre+'2010_constant_mu_sigma_scaling04.nc',
    pre+'2010_constant_mu_sigma_scaling05.nc',
    pre+'2010_constant_mu_sigma_scaling06.nc',
    pre+'2010_constant_mu_sigma_scaling07.nc',
    pre+'2010_constant_mu_sigma_scaling08.nc',
    pre+'2010_constant_mu_sigma_scaling09.nc',
    pre+'2010_constant_mu_sigma_scaling10.nc',
    pre+'2010_constant_mu_sigma_scaling11.nc',
    pre+'2010_constant_mu_sigma_scaling12.nc',
    pre+'2012_constant_mu_sigma_scaling01.nc',
    pre+'2012_constant_mu_sigma_scaling02.nc',
    pre+'2012_constant_mu_sigma_scaling03.nc',
    pre+'2012_constant_mu_sigma_scaling04.nc',
    pre+'2012_constant_mu_sigma_scaling05.nc',
    pre+'2012_constant_mu_sigma_scaling06.nc',
    pre+'2012_constant_mu_sigma_scaling07.nc',
    pre+'2012_constant_mu_sigma_scaling08.nc',
    pre+'2012_constant_mu_sigma_scaling09.nc',
    pre+'2012_constant_mu_sigma_scaling10.nc',
    pre+'2012_constant_mu_sigma_scaling11.nc',
    pre+'2012_constant_mu_sigma_scaling12.nc',
    pre+'2014_constant_mu_sigma_scaling01.nc',
    pre+'2014_constant_mu_sigma_scaling02.nc',
    pre+'2014_constant_mu_sigma_scaling03.nc',
    pre+'2014_constant_mu_sigma_scaling04.nc',
    pre+'2014_constant_mu_sigma_scaling05.nc',
    pre+'2014_constant_mu_sigma_scaling06.nc',
    pre+'2014_constant_mu_sigma_scaling07.nc',
    pre+'2014_constant_mu_sigma_scaling08.nc',
    pre+'2014_constant_mu_sigma_scaling09.nc',
    pre+'2014_constant_mu_sigma_scaling10.nc',
    pre+'2014_constant_mu_sigma_scaling11.nc',
    pre+'2014_constant_mu_sigma_scaling12.nc',
    pre+'2015_constant_mu_sigma_scaling01.nc',
    pre+'2015_constant_mu_sigma_scaling02.nc',
    pre+'2015_constant_mu_sigma_scaling04.nc',
    pre+'2015_constant_mu_sigma_scaling06.nc',
    pre+'2015_constant_mu_sigma_scaling07.nc',
    pre+'2015_constant_mu_sigma_scaling08.nc',
    pre+'2015_constant_mu_sigma_scaling09.nc',
    pre+'2015_constant_mu_sigma_scaling10.nc',
    pre+'2015_constant_mu_sigma_scaling11.nc',
    pre+'2015_constant_mu_sigma_scaling12.nc'
            ]       

test_files = [
    pre+'2015_constant_mu_sigma_scaling05.nc',
         ]

# Customized Dataloader - good for loading batches from:
# -- carefully designed input files for both single-column and nonlocal global training, and 
# -- single point single-column and nonlocal training

# tricks to speed up pytorch dataloading: https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39
# dask scheduling suggestion: https://discuss.pytorch.org/t/problems-using-dataloader-for-dask-xarray-netcdf-data/108270
#dask.config.set(scheduler='synchronous')
# ===================================================================================
# DEFINING DATALOADER
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, files, domain, stencil, batch_size, manual_shuffle, region='1andes'):

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
        
        self.nt = len(self.ds.time)
        
        self.bs = batch_size
        self.domain=domain # acceptable values: singlepoint, regional, global
        self.stencil=stencil # for nonlocal training
        self.fac = int(self.stencil/2.)
        self.manual_shuffle=manual_shuffle
        #self.index = 0
        
        if domain == 'global':
            # 122 channels for each feature
            #self.v = np.arange(0,369) # for u,v,theta
            self.v = np.arange(0,491) # for u,v,theta,w
            #self.v = np.concatenate(  (np.arange(0,247),np.arange(369,491)), axis=0) # for u,v,w
        elif domain == 'stratosphere_only':
            # 60 channels for each feature
            #self.v = np.arange(0,183) # for u,v,theta
            self.v = np.arange(0,243) # for u,v,theta,w
            #self.v = np.concatenate(  (np.arange(0,123),np.arange(183,243)), axis=0) # for u,v,w

        self.idim = len(self.v)

        # create permutations
        if self.manual_shuffle:
            self.seed  = 51
            self.tperm = np.random.RandomState(seed=self.seed).permutation(np.arange(0,self.nt))
            
        if self.domain == 'singlepoint':
            self.y0 = 14
            self.x0 = 102
        
        if self.domain == 'regional':
            #self.y1 = 10
            #self.y2 = 18
            #self.x1 = 98
            #self.x2 = 106
            if self.region == '1andes':
                self.y1=3
                self.y2=21
                self.x1=96
                self.x2=113
            if self.region == '2scand':
                self.y1=45
                self.y2=58
                self.x1=0
                self.x2=12
            if self.region == '3himalaya':
                self.y1=41
                self.y2=54
                self.x1=26
                self.x2=44
            if self.region == '4newfound':
                self.y1=47
                self.y2=58
                self.x1=103
                self.x2=119
            if self.region == '5south_ocn':
                self.y1=8
                self.y2=17
                self.x1=10
                self.x2=25
            if self.region == '6se_asia':
                self.y1=33
                self.y2=42
                self.x1=32
                self.x2=49
            if self.region == '7natlantic':
                self.y1=31
                self.y2=44
                self.x1=112
                self.x2=124
            if self.region == '8npacific':
                self.y1=27
                self.y2=47
                self.x1=67
                self.x2=87
            
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
                I = torch.from_numpy(self.inp[it,self.v,self.y0,self.x0].data.compute())
                O = torch.from_numpy(self.out[it,:,self.y0,self.x0].data.compute())
                return I,O
            else:
                # Add boundary conditions!
                I = torch.from_numpy(self.inp[it,self.v,self.y0-self.fac:self.y0+self.fac+1,self.x0-self.fac:self.x0+self.fac+1].data.compute())
                O = torch.from_numpy(self.out[it,:,self.y0,self.x0].data.compute())
                return I,O 

        elif self.domain == 'regional':
            
            if self.stencil == 1:
                I = torch.from_numpy(self.inp[it,self.v,y1:y2,x1:x2].data.compute())
                O = torch.from_numpy(self.out[it,:,y1:y2,x1:x2].data.compute())
                
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
                I = torch.from_numpy(self.inp[it,self.v,self.y1:self.y2,self.x1:self.x2,:,:].data.compute())
                O = torch.squeeze(torch.from_numpy(self.out[it,:,self.y1:self.y2,self.x1:self.x2,self.fac,self.fac].data.compute()))
                # reorder it
                I = torch.permute(I, (1,2,0,3,4))
                O = torch.permute(O, (1,2,0))
                S = I.shape
                I = I.reshape(S[0]*S[1], S[2], S[3], S[4])
                S = O.shape
                O = O.reshape(S[0]*S[1], -1)

                return I,O
 
        elif self.domain == 'global':

            if self.stencil == 1:
                I = torch.from_numpy(self.inp[it,self.v,:,:].data.compute())
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
                I = torch.from_numpy(self.inp[it,self.v,:,:,:,:].data.compute())
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
# ==================================================================================
# DEFINING THE MODEL ARCHITECTURE
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

        # Applying multiple 3x3 conv layers than just one stencilxstencil layer performs better
        if self.fac == 1:
            self.conv1 = nn.Conv2d(in_channels=idim, out_channels=idim, kernel_size=3, stride=1, padding=0)
            self.act_cnn = nn.ReLU()
            self.dropout0 = nn.Dropout(p=0.5*self.dropout_prob)
            write_log('CNN 1')
        elif self.fac == 2:
            self.conv1 = nn.Conv2d(in_channels=idim, out_channels=idim, kernel_size=3, stride=1, padding=0)
            self.act_cnn = nn.ReLU()
            self.dropout0 = nn.Dropout(p=0.5*self.dropout_prob)
            write_log('-CNN 1')
            self.conv2 = nn.Conv2d(in_channels=idim, out_channels=idim, kernel_size=3, stride=1, padding=0)
            self.act_cnn2 = nn.ReLU()
            self.dropout0_2 = nn.Dropout(p=0.5*self.dropout_prob)
            write_log('-CNN 2')
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

        if self.fac == 1:
            x = torch.squeeze( self.dropout0(self.act_cnn(self.conv1(x))) )
        elif self.fac == 2:
            x = torch.squeeze( self.dropout0(self.act_cnn(self.conv1(x))) )
            x = torch.squeeze( self.dropout0_2(self.act_cnn2(self.conv2(x))) ) 

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


# ====================================================================================================
# DEFINING THE TRAINING LOOP
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
        
        write_log(f'Epoch {epoch}, {(epoch-init_epoch+1)}/{nepochs}, training mseloss: {LOSS_TRAIN[epoch-1-init_epoch]:.6f}, testing mseloss: {LOSS_TEST[epoch-1-init_epoch]:.6f}')
        
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
# ===============================================================
# DEFINING RUN HYPERPARAMETERS AND SETTING UP THE RUN
# multiple batches from the vectorized matrices
if stencil == 1:
    bs_train=20#40#37
    bs_test=20#40#37
elif stencil > 1:
    bs_train=10
    bs_test=10

write_log(f'train batch size = {bs_train}')
write_log(f'validation batch size = {bs_test}')

# ============================= 1x1 ===========================
tstart=time2()
# IF REGIONAL, SPECIFY THE REGION AS WELL
# 1andes, 2scand, 3himalaya, 4newfound, 5south_ocn, 6se_asia, 7natlantic, 8npacific
rgn='1andes'
write_log(f'Region: {rgn}')
# shuffle=False leads to much faster reading! Since 3x3 and 5x5 is slow, set this to False
trainset1 = Dataset(files=train_files,domain='global', region=rgn, stencil=stencil, manual_shuffle=False, batch_size=bs_train)
trainloader1 = torch.utils.data.DataLoader(trainset1, batch_size=bs_train,
                                          drop_last=False, shuffle=False, num_workers=8)#, persistent_workers=True)

testset1 = Dataset(files=test_files, domain='global', region=rgn, stencil=stencil, manual_shuffle=False, batch_size=bs_test)
testloader1 = torch.utils.data.DataLoader(testset1, batch_size=bs_test,
                                         drop_last=False, shuffle=False, num_workers=8)#, persistent_workers=True)

tend=time2()
write_log(f'total_time={tend-tstart}')


# ================ 1x1 model ======================================================
idim    = trainset1.idim
odim    = trainset1.odim
hdim    = 4*idim
write_log(f'Input dim: {idim}, hidden dim: {hdim}, output dim: {odim}')
lr_min = 1e-4
lr_max = 6e-4

# lr 10-6 to 10-4 over 100 up and 100 down steps works well waise
model1     = ANN_CNN(idim=idim,odim=odim,hdim=hdim,dropout=0.2, stencil=trainset1.stencil)
model1 = model1.to(device)
write_log(f'model1 created. \n --- model1 size: {model1.totalsize():.2f} MBs,\n --- Num params: {model1.totalparams()/10**6:.3f} mil. ')
optim1     = optim.Adam(model1.parameters(),lr=1e-4)
scheduler1 = torch.optim.lr_scheduler.CyclicLR(optim1, base_lr=lr_min, max_lr=lr_max, step_size_up=50, step_size_down=50, cycle_momentum=False)

loss_fn    = nn.MSELoss()

fac = torch.ones(122)#torch.from_numpy(rho[15:]**0.1)
fac = (1./fac).to(torch.float32)
fac=fac.to(device)

print('Model created')

tstart=time2()
if domain == 'global':
    file_prefix = f"torch_saved_models/icml_global/global_{stencil}x{stencil}_era5_ann_cnn_uvthetaw_leakyrelu_dropout0p2_cyclic_mseloss" 
elif domain == 'stratosphere_only':
    file_prefix = f"torch_saved_models/stratosphere_only/ss_only_{stencil}x{stencil}_era5_ann_cnn_uvthetaw_leakyrelu_dropout0p2_cyclic_mseloss"

if restart:
    #init_epoch = 9 # epoch to start new training from, reading from (epoch-1) file
    
    idim    = trainset1.idim
    odim    = trainset1.odim
    hdim    = 4*idim
    model1     = ANN_CNN(idim=idim,odim=odim,hdim=hdim,dropout=0.2, stencil=trainset1.stencil)
    write_log(f'model1 created. \n --- model1 size: {model1.totalsize():.2f} MBs,\n --- Num params: {model1.totalparams()/10**6:.3f} mil. ')
    model1 = model1.to(device) # important to make this transfer before the optimizer step in the next line. Otherwise eror
    optim1     = optim.Adam(model1.parameters(),lr=1e-4)
    scheduler1 = torch.optim.lr_scheduler.CyclicLR(optim1, base_lr=lr_min, max_lr=lr_max, step_size_up=50, step_size_down=50, cycle_momentum=False)
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
