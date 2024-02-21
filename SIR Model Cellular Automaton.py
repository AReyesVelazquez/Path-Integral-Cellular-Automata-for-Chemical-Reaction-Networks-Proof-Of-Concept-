#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 00:01:03 2023

@author: abrahamreyes
"""
from PIL import Image
import numpy as np

def contagion(rate,inf,sus):
    return rate*inf*sus

def recovery(rate,inf):
    return rate*inf

def grayscale(A):
    pixels = np.zeros((len(A),len(A),len(A)))
    for a in range(len(A)):
        for b in range(len(A)):
            for c in range(len(A)):
                pixels[a][b][c] = 1500*A[a][b][c]
    return pixels

def proy_2d_0(S):
    proyection_0 = np.zeros(len(S))
    for a in range(len(S)):
        proyection_0[a] = sum(S[a])
    return proyection_0

def proy_2d_1(I):
    proyection_1 = np.zeros(len(I))
    for a in range(len(I)):
        for b in range(len(I)):
            proyection_1[a] += I[b][a]
    return proyection_1

def proy_3d_01(R):
    proyection_01 = np.zeros([len(R),len(R)])
    for a in range(len(R)):
        for b in range(len(R)):
            for c in range(len(R)):
                proyection_01[a][b] += R[a][b][c]
    return proyection_01

def proy_3d_01_2(R):
    proyection_01 = np.zeros([len(R),len(R)])
    for a in range(len(R)):
        for b in range(len(R)):
            for c in range(len(R)):
                proyection_01[a][b] += 10*R[a][b][c]
    return proyection_01

def proy_3d_02(R):
    proyection_02 = np.zeros([len(R),len(R)])
    for a in range(len(R)):
        for b in range(len(R)):
            for c in range(len(R)):
                proyection_02[a][b] += R[a][c][b]
    return proyection_02

def concat_horiz(im1, im2):
    dst = Image.new('L', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def concat_vert(im1, im2, im3):
    dst = Image.new("RGBA", (im1.width, im1.height + im2.height + im3.height + 61), color=(11, 76, 73))
    dst.paste(im1, (0, 31))
    dst.paste(im2, (0, im1.height + 46))
    dst.paste(im3, (0, im1.height + im2.height + 61))
    return dst

dt = 1
it = 300

N  = 100
nS = 99
nI = 1
nR = 0

R0   = 3.0
trec = 3.0

Gamm1 = (R0*dt)/(trec*N)
Gamm2 = dt/trec

InitLatt = np.zeros([N+1,N+1,N+1]) 

InitLatt[nS][nI][nR] = 1                            

rasters_nSnI  = []                                  
rasters_nS    = []                                  
rasters_nI    = []                                  
rasters_nR    = []                                   

rend = grayscale(InitLatt)
rend = rend.astype('uint32')

mat_nSnI = proy_3d_01(rend)
mat_nSnR = proy_3d_02(rend)

vec_nS  = proy_2d_0(mat_nSnI)
vec_nI  = proy_2d_1(mat_nSnI)
vec_nR  = proy_2d_1(mat_nSnR)

mat_nSnI = proy_3d_01_2(rend)

img_nSnI = Image.fromarray(mat_nSnI)
img_nS   = Image.fromarray(vec_nS)
img_nI   = Image.fromarray(vec_nI)
img_nR   = Image.fromarray(vec_nR)

img_nSnI = img_nSnI.resize((img_nSnI.width*10,img_nSnI.height*10))
img_nSnI = img_nSnI.transpose(Image.Transpose.FLIP_TOP_BOTTOM)    

rasters_nSnI.append(img_nSnI)
rasters_nS.append(img_nS)
rasters_nI.append(img_nI)
rasters_nR.append(img_nR)

NewLatt = np.zeros([N+1,N+1,N+1])                    #Lattice A Llenar

it_count = 0
while it_count < it:
    
    iterat = np.nditer(InitLatt,flags=['multi_index'], op_flags=['readonly'])
    
    while not iterat.finished:
        prob_value = InitLatt[iterat.multi_index]
        
        if prob_value > 0:    
            s,i,r = iterat.multi_index
            Rcon  = contagion(Gamm1,s,i)
            Rrec  = recovery(Gamm2,i)
            Av    = (Rcon+Rrec+1)
            Pcon  = Rcon/Av
            Prec  = Rrec/Av
            Pnad  = 1/Av
            
            NewLatt[s][i][r] += Pnad*prob_value
            
            if s-1 > 0 and i+1 < N:
                NewLatt[s-1][i+1][r] += Pcon*prob_value
            else:
                NewLatt[s][i][r] += Pcon*prob_value
                
            if i-1 > 0 and r+1 < N:
                NewLatt[s][i-1][r+1] += Prec*prob_value
            else:
                NewLatt[s][i][r] += Prec*prob_value
                
        iterat.iternext()
        
    InitLatt = NewLatt
    NewLatt  = np.zeros([N+1,N+1,N+1]) 
    rend = grayscale(InitLatt)
    rend = rend.astype('uint32')

    mat_nSnI = proy_3d_01(rend)
    mat_nSnR = proy_3d_02(rend)

    vec_nS  = proy_2d_0(mat_nSnI)
    vec_nI  = proy_2d_1(mat_nSnI)
    vec_nR  = proy_2d_1(mat_nSnR)
    
    mat_nSnI = proy_3d_01_2(rend)

    img_nSnI = Image.fromarray(mat_nSnI)
    img_nS   = Image.fromarray(vec_nS)
    img_nI   = Image.fromarray(vec_nI)
    img_nR   = Image.fromarray(vec_nR)

    img_nSnI = img_nSnI.resize((img_nSnI.width*10,img_nSnI.height*10))
    img_nSnI = img_nSnI.transpose(Image.Transpose.FLIP_TOP_BOTTOM)    

    rasters_nSnI.append(img_nSnI)
    rasters_nS.append(img_nS)
    rasters_nI.append(img_nI)
    rasters_nR.append(img_nR)
    
    it_count += 1
    print(it_count)
    
rasters_nSnI[0].save('Sir Model nS-nI Raster (Init. Delta).gif', save_all=True, append_images=rasters_nSnI[1:it], duration = 15)

for i in rasters_nS:
    temp = concat_horiz(rasters_nS[0],i)
    rasters_nS[0] = temp
temp = temp.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
temp.save('SIR Model nS Raster (Init. Delta).png')

for i in rasters_nI:
    temp2 = concat_horiz(rasters_nI[0],i)
    rasters_nI[0] = temp2
temp2 = temp2.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
temp2.save('SIR Model nI Raster (Init. Delta).png')

for i in rasters_nR:
    temp3 = concat_horiz(rasters_nR[0],i)
    rasters_nR[0] = temp3
temp3 = temp3.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
temp3.save('SIR Model nR Raster (Init. Delta).png')

temp4 = concat_vert(temp,temp2,temp3)
temp4.save('SIR Model Population Dynamics Comparison (Init. Delta).png')
