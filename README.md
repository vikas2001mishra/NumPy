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





    



               
   
