#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:21:34 2023

@author: abrahamreyes
"""
from PIL import Image
import numpy as np

def grayscale(A):
    pixels = np.zeros(len(A))
    for a in range(len(A)):
        pixels[a] = 1000*A[a]
    return pixels
        
n  = 50
rd = 1/n
dt = 1
it = 500

InitLatt = np.zeros(n+1)
arr = np.empty((0,n+1))

InitLatt[n] = 1
rend = grayscale(InitLatt)
arr = np.concatenate( ( arr, [rend] ) , axis=0)

NewLatt = np.zeros(n+1)
it_count = 0
while it_count < it:
    for i in range(n+1):
        if InitLatt[i] > 0:
            Pd = i*rd*dt
            Av = (Pd+1)
            P1 = Pd/Av
            P0 = 1/Av
            NewLatt[i]   += P0*InitLatt[i]
            if i-1 >= 0:    
                NewLatt[i-1] += P1*InitLatt[i]
            else:
                NewLatt[i] += P1*InitLatt[i]
    InitLatt = NewLatt
    NewLatt  = np.zeros(n+1)
    rend = grayscale(InitLatt)
    arr = np.concatenate( ( arr, [rend] ) , axis=0)
    it_count += 1

arr = arr.astype('uint32')
new_rend = Image.fromarray(arr)
new_rend = new_rend.transpose(Image.Transpose.ROTATE_90)

nw_rnd = Image.new("RGB",(new_rend.width,new_rend.height+60),color=(11, 76, 73))
nw_rnd.paste(new_rend, (0,30))

nw_rnd.save("Exponential Decay Population Dynamics.png")
new_rend.save('Exponential Decay Render.png')
