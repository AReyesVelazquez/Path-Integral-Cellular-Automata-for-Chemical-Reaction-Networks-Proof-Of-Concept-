#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:54:12 2023

@author: abrahamreyes
"""
from PIL import Image
import numpy as np

def grayscale(A):
    pixels = np.zeros((len(A),len(A)))
    for a in range(len(A)):
        for b in range(len(A)):
            pixels[a][b] = 1000*A[a][b]
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

def concat_horiz(im1, im2):
    dst = Image.new('L', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def concat_vert(im1, im2):
    dst = Image.new("RGBA", (im1.width, im1.height + im2.height + 46), color="pink")
    dst.paste(im1, (0, 31))
    dst.paste(im2, (0, im1.height + 46))
    return dst

dt = 1
it = 300

N   = 100                                           #Número total de partículas
n1  = 100                                           #Número de partículas en Estado 1
n0  = 0                                             #Número de partículas en Estado 0
rd  = 1/20                                       #Tasa de Decaimiento
ra  = 1/35                                         #Tasa de Absorción


InitLatt         = np.zeros([N+1,N+1])              #Lattice Inicial

InitLatt[n0][n1] = 1                                #Estado Inicial del Sistema

renders     = []                                    #Arreglo para indexar los renders
rasters_n0  = []                                    #Arreglo para indexar las proyecciones en n0
rasters_n1  = []                                    #Arreglo para indexar las proyecciones en n1

rend = grayscale(InitLatt)
rend = rend.astype('uint32')

ras0 = proy_2d_0(rend)
ras1 = proy_2d_1(rend)

mat_n0n1 = Image.fromarray(rend)
vec_n0   = Image.fromarray(ras0)
vec_n1   = Image.fromarray(ras1)

#print(ras1)

mat_n0n1 = mat_n0n1.resize((mat_n0n1.width*10,mat_n0n1.height*10))
#vec_n0 = vec_n0.resize((vec_n0.width*5,vec_n0.height*5))
#vec_n1 = vec_n1.resize((vec_n1.width*5,vec_n1.height*5))

renders.append(mat_n0n1)
rasters_n0.append(vec_n0)
rasters_n1.append(vec_n1)

NewLatt = np.zeros([N+1,N+1])                       #Lattice A Llenar
it_count = 0
while it_count < it:
    for i in range(N+1):
        for j in range(N+1):
            if InitLatt[i][j] > 0:
                Rd = j*rd
                Ra = i*ra
                Av = (Rd+Ra+1)
                Pd = Rd/Av
                Pa = Ra/Av
                Pn = 1-(Pd+Pa)
                NewLatt[i][j]     += Pn*InitLatt[i][j]
                if i+1 <= N and j-1 >= 0:
                    NewLatt[i+1][j-1] += Pd*InitLatt[i][j]
                else:
                    NewLatt[i][j] += Pd*InitLatt[i][j]
                if j+1 <= N and i-1 >= 0:
                    NewLatt[i-1][j+1] += Pa*InitLatt[i][j]
                else:
                    NewLatt[i][j] += Pa*InitLatt[i][j]
    InitLatt = NewLatt
    NewLatt  = np.zeros([N+1,N+1])
    rend = grayscale(InitLatt)
    rend = rend.astype('uint32')

    ras0 = proy_2d_0(rend)
    ras1 = proy_2d_1(rend)

    mat_n0n1 = Image.fromarray(rend)
    vec_n0   = Image.fromarray(ras0)
    vec_n1   = Image.fromarray(ras1)
    
    #print(ras1)
    
    mat_n0n1 = mat_n0n1.resize((mat_n0n1.width*10,mat_n0n1.height*10))
    #vec_n0 = vec_n0.resize((vec_n0.width*5,vec_n0.height*5))
    #vec_n1 = vec_n1.resize((vec_n1.width*5,vec_n1.height*5))

    renders.append(mat_n0n1)
    rasters_n0.append(vec_n0)
    rasters_n1.append(vec_n1)
    
    it_count += 1

renders[0].save('Birth-Death Process Dynamics Timelapse.gif', save_all=True, append_images=renders[1:it])

for i in rasters_n0:
    temp = concat_horiz(rasters_n0[0],i)
    rasters_n0[0] = temp

temp = temp.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
temp.save('Birth-Death Process n0 Population.png')


for i in rasters_n1:
    temp2 = concat_horiz(rasters_n1[0],i)
    rasters_n1[0] = temp2

temp2 = temp2.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
temp2.save('Birth-Death Process n1 Population.png')

temp4 = concat_vert(temp,temp2)
temp4.save('Birth-Death Process Population Dynamics Comparison.png')








