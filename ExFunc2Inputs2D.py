# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:47:24 2024
Example of external function with tailored autograd
This is the most simple external function, with one input vector and one output vector
@author: yoavb
"""

import torch

class CExFunc2Inputs2D(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input2D, input2):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        #ctx.save_for_backward(input)
        with torch.no_grad():
            s = input2D.shape
            nLines = s[0]
            #print(f'{input2D.shape=}')
            #print(f'{nLines=}')
            i2 = input2D.clone()
            for iLine in range(nLines):
                i2[iLine] = input2D[iLine] * 2 + input2 + iLine
        return i2

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        #input, = ctx.saved_tensors
        #return grad_output * 1.5 * (5 * input ** 2 - 1)
        #print(f'{grad_output.shape=}')
        return grad_output, None


def main():
    print('*** Check CExFunc2Inputs')
    tensor1 = torch.randn(5)
    tensor2 = torch.randn(5)
    print(f'{tensor1=}')
    print(f'{tensor2=}')
    #func = CExFunc()
    P = CExFunc2Inputs2D.apply
    res = P(tensor1, tensor2)
    print(f'{res=}')
    back = CExFunc2Inputs2D.backward(None, res)
    print(f'{back=}')
    

if __name__ == '__main__':
    main()
