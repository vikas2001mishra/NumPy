# NumPy:
    # NumPy is a Python library used for working with arrays.    
    # NumPy stands for Numerical Python.

# Advantages of Numpy:-
    # Consumes less memory.
    # Fast as the compared to python list.

# Compare Python List and NumPy Array.

                       Python List                                                                                                   NumPy Array

               # Can contain data of different data types.	                                                       # Can contain data of same data type only.
               # It is slow as compared to NumPy Array.                                                            # It is faster as compared to list.




# Dimension of array:
     -> 1D - [1 2 3 4]
     -> 2D - [[1 2 3 4]]
     -> 3D - [[[1 2 3 4]]]



# 1. Our first Program In NumPy array:->

import numpy
arr = numpy.array([1, 2, 3, 4, 5])
print(arr)


# 2.

import numpy as np
print(np.__version__)




# Create a NumPy ndarray Object:->
      -> NumPy is used to work with arrays. The array object in NumPy is called ndarray.

# 3.

import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))



# 4. Use a tuple to create a NumPy array:

import numpy as np
arr = np.array((1, 2, 3, 4, 5))
print(arr)



# 5. 0-D Arrays:-
    ->Create a (0-D) array with value 42


import numpy as np
arr = np.array(42)
print(arr)



# 6. 1-D Arrays:-
# Create a 1-D array containing the values 1,2,3,4,5:

import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(arr.ndim)


# 7. 2-D Arrays
# Create a 2-D array containing two arrays with the values 1,2,3 and 4,5,6:

import numpy as np
arr = np.array([[1, 2, 3],[4, 5, 6]])
print(arr)
print(arr.ndim)


# 8. 3-D arrays
# Create a 3-D array with two 2-D arrays, both containing two arrays with the values 1,2,3 and 4,5,6:

import numpy as np
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr)
print(arr.ndim)


# 9. n-D arrays

import numpy as np
arr = np.array([1,2,3,4],ndmin = 10)
print(arr)
print(arr.ndim)




# 10. Check Number of Dimensions?

'''import numpy as np

a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)



# 11.

import numpy as np
a = [1,2,3,4,5]
x = np.array(a)
print(x)
print(type(x))


# 12.

import numpy as np
array = np.array([1,3,5,7,9])
print(array)



# 13.

import numpy as np
l = []
for i in range(1,5):
    n = int(input("Enter the number: "))
    l.append(n)
print(np.array(l))






# Create Numpy array using NumPy functions:
  # Special NumPy array:
    # 1. Array filled with 0's
    # 2. Array filled with 1's
    # 3. Create an empty array
    # 4. An array with a range of elements
    # 5. Array diagonal element filled with 1's
    # 6. Craete an array with values that are spaced linearly in specified interval




# 1. Array filled with 0's

import numpy as np
ar_zero = np.zeros(4)
print(ar_zero)


# .

import numpy as np
ar_zero1 = np.zeros((3,4))
print(ar_zero1)




# 2. Array filled with 1's

import numpy as np
ar_one = np.ones(4)
print(ar_one)


# .

import numpy as np
ar_one = np.ones((3,4))
print(ar_one)



# 3. Create an empty array

import numpy as np
ar_empty = np.empty(4)
print(ar_empty)


# .

import numpy as np
ar_empty = np.empty((3,4))
print(ar_empty)




# 4. An array with a range of elements

import numpy as np
ar_range = np.arange(9)
print(ar_range)




# 5. Array Diagonal element filled with 1's

import numpy as np
ar_dia = np.eye(4)
print(ar_dia)

# .

import numpy as np
ar_dia = np.eye(3,5)
print(ar_dia)


 # 6. Craete an array with values that are spaced linearly in specified interval
# linspace:



import numpy as np
ar_lin = np.linspace(0,20,num = 5)
print(ar_lin)






# Create Numpy array with Random numbers:
 # Random:
   # rand()
   # randn()
   # ranf()
   # randint()



# rand(): (generate value b/w 0 to 1)

import numpy as np
var = np.random.rand(4)
print(var)


# .

import numpy as np
var = np.random.rand(4,6)
print(var)



# randn(): (generate value close to 0(zero). this may return positive or negative)

import numpy as np
var = np.random.randn(4)
print(var)

# .

import numpy as np
var = np.random.randn(3,4)
print(var)



# ranf(): (float in the half open [0.0,1.0) )

import numpy as np
f = np.random.ranf(4)
print(f)



# .

import numpy as np
f = np.random.ranf((3,4))
print(f)



# randint(): (generate a random number b/w a given range)

import numpy as np
ran_int = np.random.randint(5,20,5)  #(min,max,total_values)
print(ran_int)





# Data Types in NumPy arrays:

import numpy as np
var1 = np.array([1,2,3,4,5,6,7,8])
print("Data type:",var1.dtype)

import numpy as np
var2 = np.array([1,2,3,4,12,13,14,15])
print("Data type:",var2.dtype)

import numpy as np
var3 = np.array([1.0,2.2,3.0,4.8])
print("Data type:",var3.dtype)

import numpy as np
var4 = np.array(['a','c','m','k'])
print("Data type:",var4.dtype)

import numpy as np
var5 = np.array(['a','c','m','k',1,2,3])
print("Data type:",var5.dtype)





# Arithmetic Operation in NumPy Arrays :

# ADD:

# 1 D:
import numpy as np
var = np.array([1,2,3,4])
varadd = var + 3
print(varadd)


# 2 D:
import numpy as np
var1 = np.array([1,2,3,4])
var2 = np.array([1,2,3,4])
varadd = var1 + var2
print(varadd)


# Using function:
import numpy as np
var = np.array([3,6,9,11])
varadd = np.add(var,3)
print(varadd)


# .

import numpy as np
var1 = np.array([1,2,3,4])
var2 = np.array([1,2,3,4])
varadd2 = np.add(var1,var2)
print(varadd2)


# SUB:

# 1 D:
import numpy as np
var = np.array([1,2,3,4])
varsub = var - 3
print(varsub)


# 2 D:
import numpy as np
var1 = np.array([1,2,3,4])
var2 = np.array([1,2,3,4])
varsub = var1 - var2
print(varsub)


# Using function:

import numpy as np
var = np.array([11,22,33,44])
varsub = np.subtract(var,10)
print(varsub)

# .

import numpy as np
var1 = np.array([99,88,77,66])
var2 = np.array([11,22,33,44])
varsub2 = np.subtract(var1,var2)
print(varsub2)



# MUL:
# 1 D:
import numpy as np
var = np.array([1,2,3,4])
varmul = var * 3
print(varmul)


# 2 D:
import numpy as np
var1 = np.array([1,2,3,4])
var2 = np.array([1,2,3,4])
varmul2 = var1 * var2
print(varmul2)


# Using function:

import numpy as np
var = np.array([1,2,3,4])
varmul = np.multiply(var,2)
print(varmul)


# .

import numpy as np
var1 = np.array([9,8,7,6])
var2 = np.array([1,2,3,4])
varmul = np.multiply(var1,var2)
print(varmul)



# DIV:

# 1 D:
import numpy as np
var = np.array([1,2,3,4])
vardiv = var / 3
print(vardiv)


# 2 D:
import numpy as np
var1 = np.array([1,2,3,4])
var2 = np.array([1,2,3,4])
vardiv2 = var1 / var2
print(vardiv2)



# Using function:

import numpy as np
var = np.array([1,2,3,5])
vardiv = np.divide(var,2)
print(vardiv)'''


# .


import numpy as np
var1 = np.array([1,2,3,4])
var2 = np.array([6,7,8,9])
vardiv2 = np.divide(var1,var2)
print(vardiv2)




# MOD:
# 1 D:

import numpy as np
var = np.array([2,4,6,8])
varmod = var%3
print(varmod)

# 2 D:

import numpy as np
var1 = np.array([1,2,3,4])
var2 = np.array([1,2,3,4])
varmod2 = var1 % var2
print(varmod2)




# Using function:

import numpy as np
var = np.array([1,2])
varmod = np.mod(var,2)
print(varmod)

# .

import numpy as np
var1 = np.array([1,2])
var2 = np.array([3,4])
varmod2 = np.mod(var1,var2)
print(varmod2)


# 2D Array:

# ADD:

import numpy as np
var1 = np.array([[1,2,3,4],[1,2,3,4]])
var2 = np.array([[1,2,3,4],[1,2,3,4]])
var3 = np.array([[1,2,3,4],[1,2,3,4]])
varadd = var1 + var2 + var3
print(varadd)




# SUB:

import numpy as np
var1 = np.array([[1,2,3,4],[1,2,3,4]])
var2 = np.array([[1,2,3,4],[1,2,3,4]])
varsub = var1 - var2
print(varsub)



# MUL:

import numpy as np
var1 = np.array([[1,2,3,4],[1,2,3,4]])
var2 = np.array([[3,2,3,4],[1,2,9,4]])

varmul = var1 * var2
print(varmul)



# MIN,MAX,Sqrt,Sin,Cos,CumSum:


import numpy as np
var = np.array([90,60,30,180])
print("max value:",np.max(var))
print("min value:",np.min(var))
print("Sqrt:",np.sqrt(var))
print("Sin value:",np.sin(var))
print("Cos value:",np.cos(var))
print("cumsum:",np.cumsum(var))




# MIN & MAX,POSITION:


import numpy as np
var = np.array([99,88,77,66])
print("max value:",np.max(var),"& Position",np.argmax(var))
print("min value:",np.min(var),"& Position",np.argmin(var))





# Shape:

import numpy as np
var = np.array([[1,2],[3,4]])
print(var)
print()
print(var.shape)


# .

import numpy as np
var1 = np.array([1,2,3,4],ndmin = 4)
print(var1)
print(var1.ndim)
print()
print(var1.shape)




# ReShape:

import numpy as np
var = np.array([1,2,3,4,5,6])
print(var)
print(var.ndim)



# .

x = var.reshape(3,2)
print(x)
print(x.ndim)




# Broadcasting In Numpy Arrays:

import numpy as np
var1 = np.array([1,2,3,4])
var2 = np.array([1,2,3,4])
print( var1 + var2)


# .

import numpy as np
var1 = np.array([1,2,3])
print(var1.shape)
print(var1)

print()

var2 = np.array([[1],[2],[3]])
print(var2.shape)
print(var2)

print()

print(var1 + var2)

print()

x = np.array([[1],[2]])
print(x.shape)

print()

y = np.array([[1,2,3],[1,2,3]])
print(y.shape)

print()

print(x + y)





# Indexing and Slicing In NumPy Arrays:


# Indexing.

import numpy as np
var = np.array([3,6,8,6,9])
print(var[4])
print(var[-3])


# 2D array:

import numpy as np
var = np.array([[9,8,7,6],[1,2,3,4]])
print(var)
print(var.ndim)
print(var[0,2])
print(var[1,2])
print(var[0,-1])
print(var[1,-2])


# 3D array:

import numpy as np
var = np.array([[[1,2,3],[4,5,6]]])
print(var)
print(var.ndim)
print(var[0,1,2])




# Slicing:

# 1D:

import numpy as np
var = np.array([1,2,3,4,5,6,7,8,9])
print(var[0:4])
print(var[4:])
print(var[:5])
print(var[::2])
print(var[0:6:2])


# 2D:

import numpy as np
var = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
print(var[1,1:])




# Iterating NumPy Arrays:

# 1D:

import numpy as np
x = np.array([12,13,14,15,16,17,18,19])
print(x)
for i in  x:
    print(i)



# 2D:

import numpy as np
x = np.array([[1,2,3,4],[5,6,7,8]])
print(x)
for i in x:
    print(i)


# .


import numpy as np
x = np.array([[1,2,3,4],[5,6,7,8]])
print(x)
for i in x:
    for j in i:
        print(j)



# 3D:

import numpy as np
x = np.array([[[1,2,3],[4,5,6],[7,8,9]]])
print(x)
for i in x:
    for j in i:
        for k in j:
            print(k)



# .

import numpy as np
x = np.array([[[1,2,3],[4,5,6],[7,8,9]]])
print(x)
print(x.ndim)
for i in x:
    print(i)



# nditer function:


import numpy as np
x = np.array([[[1,2,3],[4,5,6],[7,8,9]]])
print(x)
print(x.ndim)
for i in np.nditer(x):
    print(i)





# Change the Data type:

import numpy as np
x = np.array([[[1,2,3],[4,5,6],[7,8,9]]])
print(x)
print(x.ndim)
for i in np.nditer(x,flags=['buffered'],op_dtypes=['S']):
    print(i)





# ndenumerate function:

import numpy as np
x = np.array([[[1,2,3],[4,5,6],[7,8,9]]])
print(x)
print(x.ndim)
for i,d in np.ndenumerate(x):
    print(i,d)




# Copy vs Views Numpy Python Array:

# Copy:- Make a copy, change the original array, and display both arrays.
# Views:- Make a view, change the original array, and display both arrays:



# Copy:

import numpy as np
var = np.array([1,2,3,4])
co = var.copy()
var[1] = 100    
#co[2] = 200
print("var :",var)
print("copy :",co)


# .

import numpy as np

arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42

print(arr)
print(x)




# Views: 

import numpy as np
var = np.array([1,2,3,4])
vi = var.view()
var[1] = 10
print("var : ",var)
print("view : ",vi)



# .

import numpy as np

arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
arr[0] = 42

print(arr)
print(x)






# Joining & Split NumPy Arrays Using (concatenate, stack, array_split ):

# Join array:- Joining means putting contents two or more arrays in a single array.
  
   
   # concatenate:


# 1D:

import numpy as np
var1 = np.array([1,2,3,4])
var2 = np.array([5,6,7,8])
res = np.concatenate((var1,var2))
print(res)



# axis=1:

import numpy as np
var1 = np.array([[1,2],[3,4]])
var2 = np.array([[5,6],[7,8]])
res = np.concatenate((var1,var2),axis=1)
print(res)

# axis=0

import numpy as np
var1 = np.array([[1,2],[3,4]])
var2 = np.array([[5,6],[7,8]])
res = np.concatenate((var1,var2),axis=0)
print(res)




# stack function:

import numpy as np
var1 = np.array([1,2,3,4])
var2 = np.array([5,6,7,8])
res = np.stack((var1,var2),axis=1)
print(res)

# .

import numpy as np
var1 = np.array([1,2,3,4])
var2 = np.array([5,6,7,8])
res = np.stack((var1,var2),axis=0)
print(res)

# .


import numpy as np
var1 = np.array([1,2,3,4])   # rows
var2 = np.array([5,6,7,8])
res = np.hstack((var1,var2))
print(res)

# .


import numpy as np
var1 = np.array([1,2,3,4])
var2 = np.array([5,6,7,8])   # colums
res = np.vstack((var1,var2))
print(res)


# .


import numpy as np
var1 = np.array([1,2,3,4])
var2 = np.array([5,6,7,8])      # height
res = np.dstack((var1,var2))
print(res)






# array_split:
  # spilitting breaks one array into multiple:


# 1D:

import numpy as np
x = np.array([1,2,3,4,5,6])
res = np.array_split(x,3)
print(res)


# 2D:

import numpy as np
x = np.array([[1,2],[3,4],[5,6]])
res = np.array_split(x,3)
print(res)

# .

import numpy as np
x = np.array([[1,2],[3,4],[5,6]])
res = np.array_split(x,3,axis=1)
print(res)






 # Search array:

import numpy as np
var = np.array([1,2,3,4,2,3,4,2,3])
x = np.where(var == 2)
print(x)


# .

import numpy as np
var1 = np.array([1,2,3,4,2,3,4,2,3])
y = np.where((var1%2) == 0)
print(y)

# .

import numpy as np
var1 = np.array([1,2,3,4,2,3,4,2,3])
y = np.where((var1%2) != 0)
print(y)




# Search sorted array:

import numpy as np
var = np.array([1,2,3,5,7,8,9])
res = np.searchsorted(var,4)
res1 = np.searchsorted(var,6)
print(res)
print(res1)




# sort array:

import numpy as np
var = np.array([11,20,99,5,77,8,97])
res = np.sort(var)
print(res)



# .

import numpy as np
var1 = np.array(['a','v','r','m','c','p','b'])
res1 = np.sort(var1)
print(res1)


# 2D:

import numpy as np
var1 = np.array([['a','v','r'],['m','c','p','b']])
res1 = np.sort(var1)
print(res1)

# .

import numpy as np
var = np.array([[11,20],[99,5],[77,8]])
print(np.sort(var))



# filter array:

import numpy as np
var1 = np.array(['a','v','r','m','c','p'])
f = [True,False,False,True,True,False]
res = var1[f]
print(res)
print(type(res))






# Shuffle
# Unique
# Resize
# flatten
# Ravel




# Shuffle

# 1D:

import numpy as np
var = np.array([1,2,3,4])
np.random.shuffle(var)
print(var)

# 2D:

import numpy as np
var = np.array([[1,2,3,4],[5,6,7,8]])
np.random.shuffle(var)
print(var)




# Unique

# 1D:

import numpy as np
var = np.array([1,2,1,3,4,2,5,2,6,3,6,4,7,9,4,6,8])
x = np.unique(var,return_index=True,return_counts=True)
print(x)

# 2D:

import numpy as np
var = np.array([[1,2,1,3,4,2],[5,2,6,3,6,4],[7,9,4,6,8,9]])
x = np.unique(var,return_index=True,return_counts=True)
print(x)


# 3D:

import numpy as np
var = np.array([[[1,3,2,3],[1,4,5,6]]])
x = np.unique(var)
print(x)



# Resize

import numpy as np
var = np.array([1,2,3,4,5,6])
r = np.resize(var,(2,3))
print(r)

# .

import numpy as np
var = np.array([1,2,3,4,5,6])
r = np.resize(var,(3,2))
print(r)

# .

import numpy as np
var = np.array([1,2,3,4,5,6])
r = np.resize(var,(3,3))
print(r)




# flatten


import numpy as np
var = np.array([1,2,3,4,5,6])
r = np.resize(var,(3,2))
print(r)
print("Flatten",r.flatten(order="F"))



# .


import numpy as np
var = np.array([1,2,3,4,5,6])
r = np.resize(var,(3,2))
print(r)
print("Flatten",r.flatten(order="C"))   # Output will be Ordered




 # Ravel

import numpy as np
var = np.array([1,2,3,4,5,6])
r = np.resize(var,(3,2))
print(r)
print("Ravel:",np.ravel(r,order="F"))


# .

import numpy as np
var = np.array([1,2,3,4,5,6])
r = np.resize(var,(3,2))
print(r)
print("Ravel:",np.ravel(r,order="A"))   #Ordered


# .

import numpy as np
var = np.array([1,2,3,4,5,6])
r = np.resize(var,(3,2))
print(r)
print("Ravel:",np.ravel(r,order="K"))






# Insert and Delete Arrays Functions:

# Insert

import numpy as np
var = np.array([1,2,3,4,5])
i = np.insert(var,2,40)
print(i)


# .

import numpy as np
var = np.array([1,2,3,4,5])
i = np.insert(var,(2,4),40)
print(i)

# .

import numpy as np
var = np.array([1,2,3,4,5])
i = np.insert(var,(2,4),10.7)   # Decimal(float) value does not count, count only int value
print(i)

# .


import numpy as np
var = np.array([[1,2,3],[4,5,6]])
i = np.insert(var,(2,4),100)   # Decimal(float) value does not count, count only int value
print(i)


#.

import numpy as np
var1 = np.array([[1,2,3],[4,5,6]])
i1 = np.insert(var1,2,10,axis=0)
print(i1)


# .

import numpy as np
var1 = np.array([[1,2,3],[4,5,6]])
i1 = np.insert(var1,2,10,axis=1)
print(i1)



# append:

import numpy as np
var = np.array([1,2,3,4,5])
i = np.append(var,45)
print(i)




# Delete

import numpy as np
var = np.array([1,2,3,4,5])
d = np.delete(var,2)
print(d)


# .


import numpy as np
var = np.array([1,2,3,4,5])
d = np.delete(var,(2,4))
print(d)




# Matrix Numpy Arrays:

import numpy as np
var = np.matrix([[1,2,3],[4,5,6]])
print(var)
print(type(var))


# .

import numpy as np
var1 = np.array([[1,2,3],[1,2,3]])
print(var1)
print(type(var1))

# .


import numpy as np
var1 = np.matrix([[1,2],[4,5]])
var2 = np.matrix([[2,3],[9,8]])
#print(var1+var2)
print(var1.dot(var2))





# Matrix Numpy Arrays:
     # Transpose
     # Swapaxes
     # Inverse
     # Power
     # Determinate




# Transpose:

import numpy as np
var = np.array([[1,2,3],[1,2,3]])
#print(np.transpose(var))
print(var.T)




 # Swapaxes:


import numpy as np
var = np.array([[1,2,3],[1,2,3]])
print(np.swapaxes(var,0,1))'''


# .


import numpy as np
var = np.array([[1,2,3],[1,2,3]])
print(np.swapaxes(var,axis1=0,axis2=0))'''


# 2D:

import numpy as np
var = np.array([[1,2],[3,4]])
print(var)
print(np.swapaxes(var,0,1))




# Inverse:


import numpy as np
var = np.array([[1,2],[3,4]])
print(var)
print(np.linalg.inv(var))




# Power:

# if n>0

import numpy as np
var = np.array([[1,2],[3,4]])
#print(var)
print(np.linalg.matrix_power(var,2))


# if n==0

import numpy as np
var = np.array([[1,2],[3,4]])
#print(var)
print(np.linalg.matrix_power(var,0))

# if n<0

import numpy as np
var = np.array([[1,2],[3,4]])
#print(var)
print(np.linalg.matrix_power(var,-2))


# Determinate
import numpy as np
var = np.array([[1,2],[3,4]])
#print(var)
print(np.linalg.det(var))


# .


import numpy as np
var = np.array([[1,2,4],[3,4,7],[9,8,5]])
#print(var)
print(np.linalg.det(var))























    



               
   
