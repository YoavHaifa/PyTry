# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 00:06:51 2024
PolyDL Main
Train NN on the creation of Poly Calibration Table
@author: yoavb
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys

from ExFunc import CExFunc
from ExFunc2Inputs import CExFunc2Inputs
from ExFunc2Inputs2D import CExFunc2Inputs2D

maxRadius = 260
nImages = 280

sfDevMap = 'D:/PolyDL/Data/ScoreDEV_count2_2_width260_height280_p1_dzoom2.float.rmat'



def LoadDevMap():
    print(f'Reading Dev Map {sfDevMap}...')
    image = np.memmap(sfDevMap, dtype='float32', mode='r').__array__()
    #size = image.size
    #print(f'{size}')
    image = torch.from_numpy(image.copy())
    image = image.view(-1,maxRadius)
    return image
    
def CreateModel(n_in, n_out):
    model = nn.Sequential(
    nn.Linear(n_in, n_out),
    nn.ReLU(),
    nn.Linear(n_out, n_out))
    return model

def External(y_pred):
    with torch.no_grad():
        res = CExFunc.forward(y_pred)
        return res

def Train1(model, inTab, target, loss_fn, optimizer):
    nEpochs = 2
    n_in_epoch = 20
    P2 = CExFunc.apply
    for epoch in range(nEpochs):
        for i in range (n_in_epoch):
            y_pred = model(inTab)
            res = P2(y_pred)
            loss = loss_fn(res, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')   
        print(f'{res=}')
        diff = res - target
        print(f'{diff=}')

def Train2(model, inTab, in2, target, loss_fn, optimizer):
    print('*** Train2 - external function with 2 input tensors')
    print(f'{inTab.shape=}')
    print(f'{in2.shape=}')
    nEpochs = 2
    n_in_epoch = 2
    P2 = CExFunc2Inputs.apply
    for epoch in range(nEpochs):
        for i in range (n_in_epoch):
            y_pred = model(inTab)
            res = P2(y_pred, in2)
            loss = loss_fn(res, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')   
        print(f'{res=}')
        diff = res - target
        print(f'{diff=}')

def Train2D(model, inTab2D, in2, target, loss_fn, optimizer):
    print('*** Train2D - external function with 2 input tensors')
    print(f'{inTab2D.shape=}')
    print(f'{in2.shape=}')
    nEpochs = 4
    n_in_epoch = 50
    size = inTab2D.shape
    nLines = size[0]
    nCols = size[1]
    target = target.view(-1,nCols)
    print(f'Train2D {nLines=}, {nCols=}')
    inTab1D = inTab2D.view(-1)
    P2 = CExFunc2Inputs2D.apply
    for epoch in range(nEpochs):
        for i in range (n_in_epoch):
            y_pred = model(inTab1D)
            y_pred_2D = y_pred.view(-1,nCols)
            res = P2(y_pred_2D, in2)
            #print(f'{res.shape=}')
            #print(f'{target.shape=}')
            loss = loss_fn(res, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')   
        print(f'{res=}')
        diff = res - target
        print(f'{diff=}')
    

def CheckTrain():
    n_in_cols = 5
    n_in_lines = 2
    n_out = 20
    model = CreateModel(n_in_cols*n_in_lines, n_out)
    print(f'{model=}')
    mse_loss = nn.MSELoss()
    #inTab = torch.randn(n_in, requires_grad=True)
    inTab2D = torch.randn([n_in_lines,n_in_cols], requires_grad=True)
    in2 = torch.ones(n_out, requires_grad=False)
    in1Line = torch.ones(n_in_cols, requires_grad=False)
    target = torch.randn(n_out)
    print("inTab2D: ", inTab2D)
    print("in2: ", in2)
    print("target: ", target)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #Train1(model, inTab, target, mse_loss, optimizer)
    #Train2(model, inTab, in2, target, mse_loss, optimizer)
    Train2D(model, inTab2D, in1Line, target, mse_loss, optimizer)
    sys.exit()

def main():
    CheckTrain()
    print('Poly DL')
    devMap = LoadDevMap()
    print(f'{devMap=}')
    size = devMap.size()
    print(f'{size}')
    #print(f'{model=}')
    
if __name__ == '__main__':
    main()
