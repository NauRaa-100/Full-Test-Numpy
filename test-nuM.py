
import numpy as np

"""Full Exam 1"""
"""Part | """

my_list=[5,10,15]
my_array=np.array(my_list)
print(my_array)
print('----------Seperate------------')


my_array=np.zeros((4,4))
print(my_array)
print('----------Seperate------------')

my_array=np.ones((3,2))
print(my_array)
print('----------Seperate------------')

my_array=np.full((2,3),9)
print(my_array)
print('----------Seperate------------')

my_array=np.eye(5)
print(my_array)

my_array=np.arange(0,21,5)
print(my_array)
print('----------Seperate------------')

my_array=np.linspace(0,1,6)
print(my_array)
print('----------Seperate------------')
np.random.rand(2,5)
print('----------Seperate------------')

my_array=np.random.randn(3,3)
print(my_array)
print('----------Seperate------------')

my_array=np.random.randint(10,51,size=(1,5))
print(my_array)
print('----------Seperate------------')

"""Part || """

arr = np.arange(1, 13).reshape(3, 4)


print(arr.shape)
print('----------Seperate------------')
print(arr.ndim)
print('----------Seperate------------')
print(arr.size)
print(arr.dtype)
print('----------Seperate------------')
#===========================================================
"""Part ||| 'Indexing' & 'Slicing' """

arr = np.array([10, 20, 30, 40, 50, 60])

print(arr[0])
print('----------Seperate------------')
print(arr[-1])
print('----------Seperate------------')
print(arr[1:4])
print('----------Seperate------------')

arr=np.arange(1,10).reshape(3,3)
print(arr)
print('----------Seperate------------')
print(arr[:,0])
print('----------Seperate------------')
print(arr[0,:])
print('----------Seperate------------')
arr = np.array([10,20,30,40,50,60])
print(arr[::2])
print(arr[arr > 25])
print(arr[[0,2,4]])
print('----------Seperate------------')
#=========================================================
"""Math Operation"""
arr = np.array([1, 2, 3, 4, 5])

print(arr+5)
print('----------Seperate------------')
print(arr-2)
print('----------Seperate------------')
print(arr*3)
print('----------Seperate------------')
print(arr/2)
print('----------Seperate------------')
print(arr**2)
print('----------Seperate------------')
print(np.sqrt(arr))
print('----------Seperate------------')
print(np.exp(arr))
print('----------Seperate------------')
print(np.log(arr))
print('----------Seperate------------')
print(np.sin(arr))
print('----------Seperate------------')
print(np.cos(arr))
print('----------Seperate------------')
print(np.tan(arr))
print('----------Seperate------------')
#======================================================
"""Part 4 'Statistics' """

arr = np.array([3, 7, 2, 9, 12, 15])


print(arr.mean())
print('----------Seperate------------')

print(np.median(arr))
print(arr.std())
print('----------Seperate------------')

print(arr.var())
print('----------Seperate------------')

print(np.max(arr))
print('----------Seperate------------')
print(np.min(arr))
print('----------Seperate------------')

print(arr.argmax())
print('----------Seperate------------')

print(arr.argmin())
print('----------Seperate------------')
#===========================================================
"""Part 6 'Matrix Operation' """
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

print(A.T)
print('----------Seperate------------')

print(A.dot(B))
print('----------Seperate------------')
print(np.dot(A,B))
print('----------Seperate------------')

print(np.matmul(A,B))
print('----------Seperate------------')

print(np.linalg.inv(A))
print('----------Seperate------------')

print(np.linalg.det(A))
print('----------Seperate------------')
eigvals, eigvecs = np.linalg.eig(A)
print(eigvals)
print(eigvecs)


print('----------Seperate------------')
#==========================================================

"""Part 7 'Stacking & Splitting' """
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])


print(np.concatenate([A,B],axis=0))
print('----------Seperate------------')
print(np.concatenate([A,B],axis=1))
print('----------Seperate------------')

print(np.vstack([A,B]))
print('----------Seperate------------')

print(np.hstack([A,B]))
print('----------Seperate------------')
arr = np.arange(10)
print(np.split(arr,2))
print('----------Seperate------------')
#==========================================================

"""Part 8 'Reshaping' """

arr = np.arange(1, 13)

print(arr.reshape(4,3))
print('----------Seperate------------')

print(arr.flatten())
print('----------Seperate------------')

print(arr.ravel())
print('----------Seperate------------')
#===========================================================
"""Part 9 'Broadcasting' """

A = np.array([[10,20,30],[40,50,60]])
B = np.array([1,2,3])


print(A+B)
print('----------Seperate------------')

print(A-B)
print('----------Seperate------------')

print(A*B)
print('----------Seperate------------')
#============================================================
"""Part 10 'About AI/ML' """


arr=np.random.randint(0,100,size=10)
print(arr)

arr=np.random.randn(5,5) 

A = np.random.randn(3,4)
B = np.random.randn(4,2)
print(A @ B)
print('----------Seperate------------')

C = np.random.randn(3,3)
eigvals, eigvecs = np.linalg.eig(C)
print(eigvals)
print(eigvecs)

print('----------Seperate------------')
img = np.random.randn(28,28)
flat = img.reshape(784,)
print(flat.shape)
print('----------Seperate------------')

#=============================================================
#=============================================================

"""Full Exam 2"""
""" Part 1 | Creation """


arr=np.array([2,4,6,8])
print(my_array)
print('----------Seperate------------')
print(np.zeros((3,3)))
print('----------Seperate------------')
print(np.ones((2,4)))
print('----------Seperate------------')
print(np.full((4,4),7))
print('----------Seperate------------')
print(np.eye(6))
print('----------Seperate------------')
print(np.arange(5,31,5))
print('----------Seperate------------')
print(np.linspace(0,1,8))
print('----------Seperate------------')
print(np.random.rand(3,5))
print('----------Seperate------------')
print(np.random.randn(4,4))
print('----------Seperate------------')
print(np.random.randint(100,200,size=10))
print('----------Seperate------------')
#============================================================
""" Part 2 | Attributes """

arr = np.arange(1, 21).reshape(4, 5)


print(arr.shape)
print('----------Seperate------------')
print(arr.ndim)
print('----------Seperate------------')
print(arr.size)
print('----------Seperate------------')
print(arr.dtype)
print('----------Seperate------------')
#=============================================================

""" Part 3 | Indexing & Slicing """

arr = np.array([5, 10, 15, 20, 25, 30])

print(arr[0])
print('----------Seperate------------')

print(arr[-1])
print('----------Seperate------------')

print(arr[2:4])
print('----------Seperate------------')
arr = np.arange(1,10).reshape(3,3)

print(arr)
print('----------Seperate------------')
print(arr[0,:])

print(arr[:,0])
print('----------Seperate------------')

arr=np.arange(20)
print(arr[::2])
print('----------Seperate------------')

print(arr[arr > 15])
print('----------Seperate------------')
print(arr[[5,2,0]])
#=========================================================

""" Part 4 | Math Operations """

arr = np.array([1, 2, 3, 4, 5])

print(arr+10)
print('----------Seperate------------')
print(arr-3)
print('----------Seperate------------')
print(arr*4)
print('----------Seperate------------')
print(arr/2)
print('----------Seperate------------')
print(arr**3)
print('----------Seperate------------')
print(np.sqrt(arr))
print(np.exp(arr))
print(np.log(arr))
print(np.sin(arr))
print(np.cos(arr))
print(np.tan(arr))
print('----------Seperate------------')
#============================================================
""" Part 5 | Statistics """

arr = np.array([12, 7, 9, 21, 15, 30])

print(np.mean(arr))
print('----------Seperate------------')
print(np.median(arr))
print('----------Seperate------------')
print(np.std(arr))
print('----------Seperate------------')
print(np.var(arr))
print('----------Seperate------------')
print(arr.min())
print(arr.max())
print(arr.argmax())
print('----------Seperate------------')
print(arr.argmin())
print('----------Seperate------------')
#============================================================

""" Part 6 | Matrix Operations """

A = np.array([[2,4],[6,8]])
B = np.array([[1,3],[5,7]])


print(A.T)
print(A@B)
print('----------Seperate------------')
print(np.linalg.inv(A))
print('----------Seperate------------')
print(np.linalg.det(A))
print('----------Seperate------------')

eigvals, eigvecs = np.linalg.eig(A)
print(eigvals)
print(eigvecs)
print('----------Seperate------------')
#=============================================================

""" Part 7 | Stacking & Splitting """

A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])


print(np.concatenate([A,B],axis=0))
print('----------Seperate------------')
print(np.concatenate([A,B],axis=1))
print('----------Seperate------------')
print(np.vstack([A,B]))
print('----------Seperate------------')
print(np.hstack([A,B]))
print('----------Seperate------------')
arr = np.arange(20)
print(np.split(arr,4))
print('----------Seperate------------')
#==============================================================

""" Part 8 | Reshaping """

arr = np.arange(1, 17)
arr1=arr.reshape(4,4)
print(arr1)
print('----------Seperate------------')
print(arr.flatten())
print('----------Seperate------------')
print(arr.ravel())
print('----------Seperate------------')
#=============================================================
""" Part 9 | Broadcasting """

A = np.array([[10,20,30],[40,50,60]])
B = np.array([2,4,6])


print(A+B)
print('----------Seperate------------')

print(A-B)
print('----------Seperate------------')

print(A*B)
print('----------Seperate------------')
#=============================================================

""" Part 10 | AI / ML """


arr= np.random.randint(100,size=15)
print(arr)
print('----------Seperate------------')
normalization=(arr-arr.min())/(arr.max() - arr.min())
print(normalization)
print('----------Seperate------------')
print(np.random.randn(6,6))
print('----------Seperate------------')

A =np.random.randn(3,4)
B =np.random.randn(4,2)
print(A @ B)
print('----------Seperate------------')

arr=np.random.randn(5,5)
vec , val=np.linalg.eig(arr)
print(vec)
print(val)
print('----------Seperate------------')

arr=np.random.randn(28,28)
arr1=arr.reshape(784,)
print(arr1)
print('----------Seperate------------')
#===============================================================
#===============================================================