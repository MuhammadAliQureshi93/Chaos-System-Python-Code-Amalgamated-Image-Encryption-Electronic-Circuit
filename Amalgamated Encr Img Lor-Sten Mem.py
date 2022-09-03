# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 14:11:57 2021

@author: Muhammad Ali Qureshi
"""

# Method 1 Working new USE ONLY THIS ONE
"""image encryption and decryption using two images xor and xor with three
dimensional  chaotic flow system with x y and z   """ 
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
# [...,::-1]
# Select The Path


path1 = 'E:/RESEARCH CHAOS/Research Files Chaos/Chaos Memristor Lorentz-Stenflo/1.jpg'
for file in glob.glob(path1):
    imgo1=cv2.imread(file) [...,::-1]


dsize=(256,256)
img2=cv2.resize(imgo1,dsize)
# For RGB Image


path2 = 'E:/RESEARCH CHAOS/Research Files Chaos/Chaos Memristor Lorentz-Stenflo/3.jpg'
for file in glob.glob(path2):
    imgo2=cv2.imread(file) [...,::-1]

img1=cv2.resize(imgo2,dsize)

b, g, r    = img1[:, :, 0], img1[:, :, 1], img1[:, :, 2] 

# Auto adjusting frame height and width
height = int(np.size(img1, 0))
width = int(np.size(img1, 1))
num=height*width
# t = np.arange(0.0, num, 0.01)

def lorenz(x, y, z, w, v, a=1,c=23,b=.7,r=1.5, bb=4, alp=5, bet=.02):
            '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
         '''
            x_dot = (a*(y-x ) + r*w) + bb* (alp+bet*v**2)*y
            y_dot = (c*x -y - x*z)
            z_dot = (x*y -b*z)
            w_dot = (-x - a*w)
            v_dot = y
            
            return x_dot, y_dot, z_dot, w_dot, v_dot


dt = 0.001
num_steps = num

# Need one more for the initial values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)
ws = np.empty(num_steps + 1)
vs = np.empty(num_steps + 1)

# Set initial values
xs[0], ys[0], zs[0],ws[0],vs[0] = (.1,.1,.1,.1,.1)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    x_dot, y_dot, z_dot, w_dot, v_dot = lorenz(xs[i], ys[i], zs[i],ws[i],vs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)
    ws[i + 1] = ws[i] + (w_dot * dt)
    vs[i + 1] = vs[i] + (v_dot * dt)

xr=xs;yr=ys;zr=zs
# do some fancy 3D plotting
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(state[:,0],state[:,1])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# # ax.set_zlabel('z')
# plt.show()

# xr=state[:,0];yr=state[:,1];zr=state[:,2]
# Data Slicing and Reshaping for Encryption
xr1=np.array(xr[0:num]*10000);yr1=np.array(yr[0:num]*10000);zr1=np.array(zr[0:num]*10000)
rr=r;bb=b;gg=g
xr2=abs(np.int16(xr1));yr2=abs(np.int16(yr1));zr2=abs(np.int16(zr1))
xrr=xr2.reshape(height,width)
yrr=yr2.reshape(height,width)
zrr=zr2.reshape(height,width)

# imgtest=imgenc^img2 #two images and chotic data

"""Encryption Portion"""
R=np.bitwise_xor(rr,xrr)
G=np.bitwise_xor(gg,yrr)
B=np.bitwise_xor(bb,zrr)
R1=R.astype(np.uint8)
G1=G.astype(np.uint8)
B1=B.astype(np.uint8)

imgenc = cv2.merge((R1, G1, B1))# one image and chotic data

imgtest=imgenc^img2 #two images and chotic data

# bt, gt, rt    = imgtest[:, :, 0], imgtest[:, :, 1], imgtest[:, :, 2] 


# plt.imshow(imgenc)
# plt.title('Daata and image')
# plt.show()

# plt.imshow(imgtest)
# plt.title('data and two images')
# plt.show()


path = 'E:/RESEARCH CHAOS/Research Files Chaos/Chaos Memristor Lorentz-Stenflo/1.jpg'
for file in glob.glob(path):
    imgo=cv2.imread(file)[...,::-1]

dsize=(256,256)
img3=cv2.resize(imgo,dsize)
"""Decryption Portion"""
imgdec2=imgtest^img3 #change her for good dec img1  and for bad img2
bt, gt, rt    = imgdec2[:, :, 0], imgdec2[:, :, 1], imgdec2[:, :, 2] 

R11=np.bitwise_xor(bt,xrr)
G11=np.bitwise_xor(gt,yrr)
B11=np.bitwise_xor(rt,zrr)
dR=R11.astype(np.uint8)
dG=G11.astype(np.uint8)
dB=B11.astype(np.uint8)



imgdec = cv2.merge((dR, dG, dB))[...,::-1]
# imgdec=imgdec0^img2

dsize=(7200,4800)
output0 = cv2.resize(img1, dsize)
output1 = cv2.resize(imgtest, dsize)
output2 = cv2.resize(imgdec, dsize)

"""ploting Portion with Histogram"""
fig, axs = plt.subplots(2,3, figsize=(20, 10))
# axs[0,0].set_title('Original Image')
axs[0,0].imshow(img1)
axs[1,0].set_title('Original Image Histogram')
axs[1,0].set_xlabel('RGB Channel')
axs[1,0].set_ylabel('No of Pixels')
axs[1,0].hist(img1.ravel(),256,[0,256])
# axs[0,1].set_title('Encrypted Image')
axs[0,1].imshow(imgtest)
axs[1,1].set_title('Encrypted Image Histogram')
axs[1,1].set_xlabel('RGB Channel')
axs[1,1].set_ylabel('No of Pixels')
axs[1,1].hist(imgtest.ravel(),256,[0,256])
# axs[0,2].set_title('Decrypted Image')
axs[0,2].imshow(imgdec)
axs[1,2].set_title('Decrypted Image Histogram')
axs[1,2].set_xlabel('RGB Channel')
axs[1,2].set_ylabel('No of Pixels')
axs[1,2].hist(imgdec.ravel(),256,[0,256])
"""
# plotting individual images for security analysis
file0='C:/Users/Muhammad Ali Qureshi/Desktop/o duff.jpg'
file1='C:/Users/Muhammad Ali Qureshi/Desktop/e duff.jpg'
file2='C:/Users/Muhammad Ali Qureshi/Desktop/d duff.jpg'


plt.figure(0)
# cv2.imshow('original',output0)
cv2.imwrite('C:/Users/Muhammad Ali Qureshi/Desktop/o duff.jpg', img1)
# plt.savefig(img1,'C:/Users/Muhammad Ali Qureshi/Desktop/o duff.eps')

# plt.axis('off')
# plt.savefig('o duff.png',dpi=200)

plt.figure(1)
# cv2.imshow('encr',output1)
cv2.imwrite('C:/Users/Muhammad Ali Qureshi/Desktop/e duff.jpg', imgtest)
# plt.savefig(imgtest,'C:/Users/Muhammad Ali Qureshi/Desktop/e duff.eps')

            
# plt.axis('off')
# plt.savefig('e duff.png',dpi=200)

plt.figure(2)
# cv2.imshow('decr',output2)
cv2.imwrite('C:/Users/Muhammad Ali Qureshi/Desktop/d duff.jpg', imgdec)
# plt.savefig(imgdec,'C:/Users/Muhammad Ali Qureshi/Desktop/d duff.eps')

# plt.axis('off')
# plt.savefig('d duff.png',dpi=200)


plt.figure(0)
a1=plt.hist(img1.ravel(),256,[0,256])
# plt.title('Original Image Histogram')
plt.xlabel('RGB Channel')
plt.ylabel('No of Pixels')
plt.savefig('C:/Users/Muhammad Ali Qureshi/Desktop/Histogram org.jpg',dpi=1200)
# plt.savefig('C:/Users/Muhammad Ali Qureshi/Desktop/Histogram org.eps',dpi=1200)

plt.figure(1)
a2=plt.hist(imgtest.ravel(),256,[0,256])
# plt.title('Encrypted Image Histogram')
plt.xlabel('RGB Channel')
plt.ylabel('No of Pixels')
plt.savefig('C:/Users/Muhammad Ali Qureshi/Desktop/Histogram encr.jpg',dpi=1200)
# plt.savefig('C:/Users/Muhammad Ali Qureshi/Desktop/Histogram encr.eps',dpi=1200)

plt.figure(3)
a3=plt.hist(imgdec.ravel(),256,[0,256])
# plt.title('Decrypted Image Histogram')
plt.xlabel('RGB Channel')
plt.ylabel('No of Pixels')
plt.savefig('C:/Users/Muhammad Ali Qureshi/Desktop/Histogram decr.jpg',dpi=1200)
# plt.savefig('C:/Users/Muhammad Ali Qureshi/Desktop/Histogram decr.eps',dpi=1200)
"""