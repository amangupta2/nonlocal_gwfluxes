import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# For ANNs and ANN+CNNs
def Training_ANN_CNN(nepochs,model,optimizer,loss_fn,trainloader,testloader,stencil, bs_train,bs_test,save,
             file_prefix, device, log_filename, init_epoch=1, scheduler=0):

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

        log = open(log_filename,'a')
        print(f'Epoch {epoch}, {(epoch-init_epoch+1)}/{nepochs}, training mseloss: {LOSS_TRAIN[epoch-1-init_epoch]:.6f}, testing mseloss: {LOSS_TEST[epoch-1-init_epoch]:.6f}', file=log)

        # Saving the model at any given epoch
        if save:
            savepath = f'{file_prefix}_train_epoch{epoch}.pt'
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


def model_freeze_transfer_learning(model, model_type):

   # freezes all but last output layers for the respective models 

    for params in model.parameters():
        params.requires_grad=False

    if model_type=='ann':
        model.output.weight.requires_grad = True
        model.output.bias.requires_grad   = True
    elif model_type=='attention':
        model.conv1x1.weight.requires_grad = True
        model.conv1x1.bias.requires_grad   = True

    return model

# For Attention UNet - reshaping etc is not needed
def Training_AttentionUNet(nepochs,model,optimizer,loss_fn,trainloader,testloader, bs_train,bs_test,save,
             file_prefix, device, log_filename, init_epoch=1, scheduler=0):

    LOSS_TRAIN = np.zeros((nepochs))
    LOSS_TEST   = np.zeros((nepochs))

    print("Training ...")
    for epoch in np.arange(init_epoch + 0, init_epoch + nepochs):

        # --------- training ----------
        #model.train()
        trainloss=0.
        count=0.
        for i, (inp, out) in enumerate(trainloader):
            inp=inp.to(device)
            out=out.to(device)
            pred   =model(inp)
            loss     = loss_fn(pred,out)
            optimizer.zero_grad()
            # backward propagation
            loss.backward()
            optimizer.step()
            if scheduler !=0:
                scheduler.step()
            trainloss += loss#.item()#.item()
            count+=1

        LOSS_TRAIN[epoch-1-init_epoch] = trainloss/count

        #--------- testing ------------
        model.eval()
        #print('===== TESTING ============')
        testloss=0.
        count=0.
        for i, (inp, out) in enumerate(testloader):
            inp=inp.to(device)
            out=out.to(device)
            pred   =model(inp)
            loss2     = loss_fn(pred,out)
            testloss += loss2.item()
            count+=1

        LOSS_TEST[epoch-1-init_epoch] = testloss/count

        log = open(log_filename,'a')
        print(f'Epoch {epoch}, {(epoch-init_epoch+1)}/{nepochs}, training mseloss: {LOSS_TRAIN[epoch-1-init_epoch]:.6f}, testing mseloss: {LOSS_TEST[epoch-1-init_epoch]:.6f}', file=log)

        # Saving the model at any given epoch
        if save:
            savepath = f'{file_prefix}_train_epoch{epoch}.pt'
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




def Training_ANN_CNN_TransferLearning(nepochs,model,optimizer,loss_fn,trainloader,testloader,stencil, bs_train,bs_test,save,
             file_prefix, device, log_filename, init_epoch=1, scheduler=0):

    LOSS_TRAIN = np.zeros((nepochs))

    for epoch in np.arange(init_epoch + 0, init_epoch + nepochs):
        # --------- training ----------
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

        log = open(log_filename,'a')
        print(f'Epoch {epoch}, {(epoch-init_epoch+1)}/{nepochs}, training mseloss: {LOSS_TRAIN[epoch-1-init_epoch]:.6f}', file=log)

        # Saving the model at any given epoch
        if save:
            savepath = f'{file_prefix}_train_epoch{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn,
                'loss_train': LOSS_TRAIN,
                #'loss_test': LOSS_TEST,
                'scheduler' : 'CyclicLR'
                }, savepath)

    return model, LOSS_TRAIN



# Differs from regular training in 1. No validation since low IFS data, 2. No model.train() is invoked, since model freeze function is invoked in the main file
# Yet, accepting testloader as an argument in case needed in the future
def Training_AttentionUNet_TransferLearning(nepochs,model,optimizer,loss_fn,trainloader,testloader, bs_train,bs_test,save,
             file_prefix, device, log_filename, init_epoch=1, scheduler=0):

    LOSS_TRAIN = np.zeros((nepochs))

    for epoch in np.arange(init_epoch + 0, init_epoch + nepochs):

        # --------- training ----------
        trainloss=0.
        count=0.
        for i, (inp, out) in enumerate(trainloader):
            inp=inp.to(device)
            out=out.to(device)
            pred   =model(inp)
            loss     = loss_fn(pred,out)
            optimizer.zero_grad()
            # backward propagation
            loss.backward()
            optimizer.step()
            if scheduler !=0:
                scheduler.step()
            trainloss += loss#.item()#.item()
            count+=1

        LOSS_TRAIN[epoch-1-init_epoch] = trainloss/count

        log = open(log_filename,'a')
        print(f'Epoch {epoch}, {(epoch-init_epoch+1)}/{nepochs}, training mseloss: {LOSS_TRAIN[epoch-1-init_epoch]:.6f}', file=log)

        # Saving the model at any given epoch
        if save:
            savepath = f'{file_prefix}_train_epoch{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn,
                'loss_train': LOSS_TRAIN,
                #'loss_test': LOSS_TEST,
                'scheduler' : 'CyclicLR'
                }, savepath)

    return model, LOSS_TRAIN
