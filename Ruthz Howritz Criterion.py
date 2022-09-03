# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 11:02:09 2022

@author: Muhammad Ali Qureshi
"""


import numpy as np
a=1;b=.7;c=-100;r=1.5; bb=4; alp=20; bet=.02 ; q=.98

a0=1; a1=2*a+b+1
a2=a**2 + 2*a*b**2 -a*c + 2*a + b - 80*c + r
a3= a**2*b - a**2*c + a**2 -a*b*c +2*a*b -80*a*c -80*b*c +b*r +r
a4= -a**2*b*c + a**2*b -80*a*b*c +b*r
a5=a6=a7=a8=a9=0
del0=a1
print(del0)

# creating a 2X2 Numpy matrix
del2 = np.array  ( [[a1, a3 ],
                    [a0, a2]])
# Displaying the Matrix  
# calculating the determinant of matrix
det = np.linalg.det(del2) 
print("\nDeterminant of given 2X2 square matrix:")
print((det))


# creating a 3X3 Numpy matrix
del3 = np.array(   [[a1, a3, a5],
                    [a0, a2, a4],
                    [0, a1, a3]])
# Displaying the Matrix  
# calculating the determinant of matrix
det = np.linalg.det(del3) 
print("\nDeterminant of given 3X3 square matrix:")
print((det))

# creating a 4X4 Numpy matrix
del4 = np.array   ([[a1, a3, a5, a7],
                    [a0, a2, a4, a6],
                    [0, a1, a3, a5],
                    [0, a0, a2, a4]])
# Displaying the Matrix  
# calculating the determinant of matrix
det = np.linalg.det(del4) 
print("\nDeterminant of given 4X4 square matrix:")
print((det))

# creating a 5X5 Numpy matrix
del5 = np.array([[a1, a3, a5, a7, a9],
                    [a0, a2, a4, a6, a8],
                    [0, a1, a3, a5, a7],
                    [0, a0, a2, a4, a6],
                    [0, 0, a1, a3, a5]])
# Displaying the Matrix  
# calculating the determinant of matrix
det = np.linalg.det(del5) 
print("\nDeterminant of given 5X5 square matrix:")
print((det))
#%%
import numpy as np

a= 24; b=4; c = 19; d=9 ;e=0

a0=1; a1=a+d
a2=a*d-a*c-b*c - .02*b*c*e**2
a3= -(.02*b*c*d*e**2+a*c*d +b*c*d)
a4=a5=a6=a7=0
del0=a1
print(del0)

# creating a 2X2 Numpy matrix
del2 = np.array  ( [[a1, a3 ],
                    [a0, a2]])
# Displaying the Matrix  
# calculating the determinant of matrix
det = np.linalg.det(del2) 
print("\nDeterminant of given 2X2 square matrix:")
print((det))


# creating a 3X3 Numpy matrix
del3 = np.array(   [[a1, a3, a5],
                    [a0, a2, a4],
                    [0, a1, a3]])
# Displaying the Matrix  
# calculating the determinant of matrix
det = np.linalg.det(del3) 
print("\nDeterminant of given 3X3 square matrix:")
print((det))

# creating a 4X4 Numpy matrix
del4 = np.array   ([[a1, a3, a5, a7],
                    [a0, a2, a4, a6],
                    [0, a1, a3, a5],
                    [0, a0, a2, a4]])
# Displaying the Matrix  
# calculating the determinant of matrix
det = np.linalg.det(del4) 
print("\nDeterminant of given 4X4 square matrix:")
print((det))

