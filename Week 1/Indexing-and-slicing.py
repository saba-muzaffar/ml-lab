#indexing and slicing
import numpy as np
a = np.array([10,20,30,40,50])
print(a[0])
print(a[-1])
print(a[1:4])
print(a[::2])

b =np.array([[1,2,3,4],[5,6,7,8],[9,1,2,3]])
print(b[0,1])
print(b[:,1])
print(b[1,:])
print(b[::,1])