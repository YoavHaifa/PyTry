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


def Train(model, inTab, target, loss_fn, optimizer):
    n = 20
    n_in_epoch = 20
    P2 = CExFunc.apply
    for epoch in range(n):
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
    

def CheckTrain():
    n_in = 4000
    n_out = 8000
    model = CreateModel(n_in, n_out)
    print(f'{model=}')
    mse_loss = nn.MSELoss()
    inTab = torch.randn(n_in, requires_grad=True)
    target = torch.randn(n_out)
    print("inTab: ", inTab)
    print("target: ", target)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    Train(model, inTab, target, mse_loss, optimizer)
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
