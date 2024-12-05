# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:47:24 2024
Example of external function with tailored autograd
@author: yoavb
"""

import torch

class CExFunc(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        #ctx.save_for_backward(input)
        with torch.no_grad():
            i2 = input * 2
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
        return grad_output


def main():
    print('*** Check CExFunc')
    tensor = torch.randn(5)
    print(f'{tensor=}')
    #func = CExFunc()
    t2 = CExFunc.forward(tensor)
    print(f'{t2=}')
    t2b = CExFunc.backward(t2)
    print(f'{t2b=}')
    

if __name__ == '__main__':
    main()
