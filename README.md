# NUMPY
#NUMERICAL PYTHON 
#numerical computation
#We are going to use Numpy in Machine Learning, Deep Learning and Computer Vision because Computer understands numbers. 
#There are lot of functions (2000+ converted from MATLAB) inside numpy but majorly we will use 10-20 functions only
import numpy as np#importing numpy lib
np.array([2,3,4,5,6])#1 Dimension
array([2, 3, 4, 5, 6])
type(np.array([2,3,4,5,6]))#checked type 
numpy.ndarray
np.array([[2,3,4] , [4,5,6]])
array([[2, 3, 4],
       [4, 5, 6]])
np.array([[1,2,3,4] , [ 2,3,4,5]])
array([[1, 2, 3, 4],
       [2, 3, 4, 5]])
l = [[1,2,3,4] , [ 2,3,4,5]]#list
l[0]
[1, 2, 3, 4]
l[0][0]
1
a = np.array([3,4,5,6] , ndmin= 4)
a
array([[[[3, 4, 5, 6]]]])
arr = np.array([[1,2,3] , [4,5,6] , [7,8,9]])
arr
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
arr.ndim
2
#Generating Random Numbers between 2 and 4
print(np.random.randint(2,5))
#Generating Random Numbers between 2 and 49
print(np.random.randint(2,50))
#Generating Multiple Random Numbers
print(np.random.randint(2,50,(3,4))) #3 Rows and 4 Columns
#Generating Multiple Random Numbers
print(np.random.randint(2,50,(2,3,4))) #3 Rows and 4 Columns and 2 such matrices
#Images are actually 3D Arrays
4
5
[[32 10  6 29]
 [20  6 33 31]
 [20 18 16 13]]
[[[ 8  8 12 47]
  [37 20 11 20]
  [49 48 13 11]]

 [[35 28 23 22]
  [46  9 20 35]
  [46 25  8 24]]]
np.random.randint(1,10,(5,4))#Random Numbers between 1 and 10, 5 Rows and 4 Columns
array([[8, 1, 2, 8],
       [1, 3, 3, 2],
       [4, 3, 6, 8],
       [4, 4, 2, 4],
       [7, 4, 3, 3]])
np.random.randint(1,10,(4,3,2))#Random Numbers between 1 and 10,4 matrices,  3 Rows and 2 Columns
array([[[9, 6],
        [7, 2],
        [1, 4]],

       [[4, 9],
        [5, 1],
        [9, 9]],

       [[5, 2],
        [3, 2],
        [4, 4]],

       [[6, 4],
        [9, 1],
        [8, 2]]])
l = [1,2,3,4,5,6]#list created
np.array(l)#passed list from array
array([1, 2, 3, 4, 5, 6])
l1 = [4,5,6,7,"sudh" , 34.565, True]##Converted everything into string automatically. String occupies higher space than boolean,integer or float values. 
#U32 is internal representation of numpy array: 32 bits, 16 bits, 64 bits etc.
np.array(l1)
array(['4', '5', '6', '7', 'sudh', '34.565', 'True'], dtype='<U32')
a1=np.array(l1)
a2 = np.array([[1,2,3] , [4,5,6]])
a2
array([[1, 2, 3],
       [4, 5, 6]])
a3 = np.array([[[1,2,3] , [4,5,6] , [7,8,9]]])
a3
array([[[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]])
#Dimensions of Array: How much axes is needed for data to be represented: When we perform data manipulation/computation, these dimensions will become important (Matrix Multiplication etc)
print(a1.ndim) #One Axis needed to represent this entire data point
print(a2.ndim) #Two Axes needed to represent this entire data point
print(a3.ndim) #Three Axes needed to represent this entire data point
1
2
3
a1.ndim#1d
1
a2.ndim#2d
2
a3.ndim#3d
3
a1
array(['4', '5', '6', '7', 'sudh', '34.565', 'True'], dtype='<U32')
a1.size
7
a2
array([[1, 2, 3],
       [4, 5, 6]])
a2.size
6
a3
array([[[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]])
a3.size
9
#Size always tells us number of elements in the array
print(a1.size)
print(a2.size)
print(a3.size)
     
7
6
9
a1
array(['4', '5', '6', '7', 'sudh', '34.565', 'True'], dtype='<U32')
#Shape gives rows and columns
print(a1.shape) 
print(a2.shape) #2 Rows and 3 Columns
print(a3.shape) #3 Rows and 3 Columns   
#We read as we have 3 cross 3 and have one such matrice (z Axis Depth,Rows,Columns)
(7,)
(2, 3)
(1, 3, 3)
a1.shape
(7,)
a2
array([[1, 2, 3],
       [4, 5, 6]])
a2.shape
(2, 3)
a3
array([[[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]])
a3.shape
(1, 3, 3)
#Decimals also: Random Function is used for Data Normalization
np.random.rand(5,4)
array([[0.3876431 , 0.35763401, 0.49325461, 0.33270194],
       [0.19140233, 0.06753103, 0.32090523, 0.28478995],
       [0.84667626, 0.01208437, 0.2242166 , 0.50774804],
       [0.29588933, 0.06202619, 0.11941257, 0.97587137],
       [0.9038082 , 0.2276159 , 0.34729487, 0.51191139]])
np.random.randn(5,4)#
#Normal Distribtuion 
array([[-1.39782617,  1.3529308 ,  2.23799923, -0.742564  ],
       [ 0.51291731, -0.82211014,  1.12794451,  0.59868766],
       [-1.69991443, -0.48779123, -0.10333153,  1.71229602],
       [-0.37326131, -0.57047055,  0.3267387 , -0.38251041],
       [-0.78576485,  1.1475976 , -0.75916029, -0.39473587]])
a4=np.random.rand(5,4)
a4
array([[0.18009131, 0.48443023, 0.75153323, 0.79041423],
       [0.64771994, 0.33088539, 0.01439887, 0.20264494],
       [0.7739492 , 0.07970932, 0.25579202, 0.25126012],
       [0.15813887, 0.9995731 , 0.35234541, 0.84031972],
       [0.21350468, 0.68903829, 0.89280891, 0.50231763]])
a4.reshape(10,2)
##Changing Shape of Data (In Images, we crop the image)
#a4.reshape(3,4) #Error because this will accomodate 12 dataset points but we have 16 dataset points
array([[0.18009131, 0.48443023],
       [0.75153323, 0.79041423],
       [0.64771994, 0.33088539],
       [0.01439887, 0.20264494],
       [0.7739492 , 0.07970932],
       [0.25579202, 0.25126012],
       [0.15813887, 0.9995731 ],
       [0.35234541, 0.84031972],
       [0.21350468, 0.68903829],
       [0.89280891, 0.50231763]])
a4.reshape(2,10)
array([[0.18009131, 0.48443023, 0.75153323, 0.79041423, 0.64771994,
        0.33088539, 0.01439887, 0.20264494, 0.7739492 , 0.07970932],
       [0.25579202, 0.25126012, 0.15813887, 0.9995731 , 0.35234541,
        0.84031972, 0.21350468, 0.68903829, 0.89280891, 0.50231763]])
a5=np.random.rand(4,4)
b1 = a5.reshape(2,8) #2 rows and 8 columns
b2 = a5.reshape(8,2) #8 rows and 2 columns
b3 = a5.reshape(1,16) #1 row and 16 columns
b4 = a5.reshape(16,1) #16 columns and 1 row

print(b1)
print(b2)
print(b3)
print(b4)
[[0.85718828 0.00906244 0.21099691 0.90557587 0.06325078 0.03561213
  0.94613534 0.14080027]
 [0.24420947 0.57311055 0.38893711 0.30553878 0.71328334 0.37447934
  0.81088702 0.890711  ]]
[[0.85718828 0.00906244]
 [0.21099691 0.90557587]
 [0.06325078 0.03561213]
 [0.94613534 0.14080027]
 [0.24420947 0.57311055]
 [0.38893711 0.30553878]
 [0.71328334 0.37447934]
 [0.81088702 0.890711  ]]
[[0.85718828 0.00906244 0.21099691 0.90557587 0.06325078 0.03561213
  0.94613534 0.14080027 0.24420947 0.57311055 0.38893711 0.30553878
  0.71328334 0.37447934 0.81088702 0.890711  ]]
[[0.85718828]
 [0.00906244]
 [0.21099691]
 [0.90557587]
 [0.06325078]
 [0.03561213]
 [0.94613534]
 [0.14080027]
 [0.24420947]
 [0.57311055]
 [0.38893711]
 [0.30553878]
 [0.71328334]
 [0.37447934]
 [0.81088702]
 [0.890711  ]]
a4.reshape(2,-1) #Since 2 is a factor of 16, -1 or any negative number means that it will 
#automatically detect the other factor which is 8 in this case
  
array([[0.18009131, 0.48443023, 0.75153323, 0.79041423, 0.64771994,
        0.33088539, 0.01439887, 0.20264494, 0.7739492 , 0.07970932],
       [0.25579202, 0.25126012, 0.15813887, 0.9995731 , 0.35234541,
        0.84031972, 0.21350468, 0.68903829, 0.89280891, 0.50231763]])
a1
array(['4', '5', '6', '7', 'sudh', '34.565', 'True'], dtype='<U32')
a1[2:6]
array(['6', '7', 'sudh', '34.565'], dtype='<U32')
a1[2:6:2]
array(['6', 'sudh'], dtype='<U32')
a1[::-1]
array(['True', '34.565', 'sudh', '7', '6', '5', '4'], dtype='<U32')
#Slicing Exact similar to list
print(a1)
print(a1[0])     
print(a1[2:6])
print(a1[2:6:2])
['4' '5' '6' '7' 'sudh' '34.565' 'True']
4
['6' '7' 'sudh' '34.565']
['6' 'sudh']
a2
array([[1, 2, 3],
       [4, 5, 6]])
a2[0]
array([1, 2, 3])
a2[:,1:]
array([[2, 3],
       [5, 6]])
a2[:,[1,2]]
array([[2, 3],
       [5, 6]])
a2[[0,1],1:]
array([[2, 3],
       [5, 6]])
print(a2)
print(a2[0])
print(a2[:,1:])
print(a2[:,[1,2]]) #Same thing as above
print(a2[[0,1],[1,2]]) #Same thing as above
[[1 2 3]
 [4 5 6]]
[1 2 3]
[[2 3]
 [5 6]]
[[2 3]
 [5 6]]
[2 6]
a5 = np.random.randint(2,90 , (6,5))
a5
array([[41, 89, 26, 22, 51],
       [58, 47, 52, 50, 75],
       [23, 17,  6, 69, 39],
       [34, 39,  8, 48, 42],
       [22, 75, 68, 35, 35],
       [25, 16, 67, 66, 76]])
##Similar to Pandas Dataframe, we can write filter condition
a5[a5>40]
array([ 41, 100,  51,  58,  47,  52,  50,  75,  69,  48,  42,  75,  68,
        67,  66,  76])
a5
array([[41, 89, 26, 22, 51],
       [58, 47, 52, 50, 75],
       [23, 17,  6, 69, 39],
       [34, 39,  8, 48, 42],
       [22, 75, 68, 35, 35],
       [25, 16, 67, 66, 76]])
#Updating data
a5[0,1] = 100#11 Replaced by 100
a5
array([[ 41, 100,  26,  22,  51],
       [ 58,  47,  52,  50,  75],
       [ 23,  17,   6,  69,  39],
       [ 34,  39,   8,  48,  42],
       [ 22,  75,  68,  35,  35],
       [ 25,  16,  67,  66,  76]])
a6 = np.random.randint(0,3 , (3,3))
a7 = np.random.randint(0,3 , (3,3))
a6
array([[2, 1, 2],
       [2, 0, 1],
       [0, 0, 0]])
a7
array([[1, 2, 1],
       [0, 2, 1],
       [1, 2, 2]])
a6*a7##Element Wise Multiplication and not Matrix Multiplication
array([[2, 2, 2],
       [0, 0, 1],
       [0, 0, 0]])
#Matrix Multiplication
a6@a7
array([[ 4, 10,  7],
       [ 3,  6,  4],
       [ 0,  0,  0]])
a6+100
array([[102, 101, 102],
       [102, 100, 101],
       [100, 100, 100]])
a6*2
array([[4, 2, 4],
       [4, 0, 2],
       [0, 0, 0]])
a6/0
C:\Users\Swapnil\AppData\Local\Temp\ipykernel_17640\1058679823.py:1: RuntimeWarning: divide by zero encountered in true_divide
  a6/0
C:\Users\Swapnil\AppData\Local\Temp\ipykernel_17640\1058679823.py:1: RuntimeWarning: invalid value encountered in true_divide
  a6/0
array([[inf, inf, inf],
       [inf, nan, inf],
       [nan, nan, nan]])
a6
array([[2, 1, 2],
       [2, 0, 1],
       [0, 0, 0]])
a6**3
array([[8, 1, 8],
       [8, 0, 1],
       [0, 0, 0]], dtype=int32)
a8 = np.zeros((4,4))
a8
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
a9  = np.ones((4,5))
a9
array([[1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]])
a9+np.array([1,2,3,4,5])#Doing row by row operation. Also called as Broadcasting operation
array([[2., 3., 4., 5., 6.],
       [2., 3., 4., 5., 6.],
       [2., 3., 4., 5., 6.],
       [2., 3., 4., 5., 6.]])
a9+10
array([[11., 11., 11., 11., 11.],
       [11., 11., 11., 11., 11.],
       [11., 11., 11., 11., 11.],
       [11., 11., 11., 11., 11.]])
##How to do column wise Boradcasting operation: We will have to transpose first and then add
#np.array([1,2,3,4]).T + a9   #Error because for transpose we need 2 dimesnsions. Hence we will give 2 brackets to convert 
#into 2d

np.array([[1,2,3,4]]).T + a9
array([[2., 2., 2., 2., 2.],
       [3., 3., 3., 3., 3.],
       [4., 4., 4., 4., 4.],
       [5., 5., 5., 5., 5.]])
#Understanding Transpose
print(a6)
print(a6.T)
[[2 1 2]
 [2 0 1]
 [0 0 0]]
[[2 2 0]
 [1 0 0]
 [2 1 0]]
np.array([[1,2,3,4]]).T
array([[1],
       [2],
       [3],
       [4]])
np.array([[1,2,3,4]])
array([[1, 2, 3, 4]])
a6.T
array([[2, 2, 0],
       [1, 0, 0],
       [2, 1, 0]])
a6
array([[2, 1, 2],
       [2, 0, 1],
       [0, 0, 0]])
a5
array([[ 41, 100,  26,  22,  51],
       [ 58,  47,  52,  50,  75],
       [ 23,  17,   6,  69,  39],
       [ 34,  39,   8,  48,  42],
       [ 22,  75,  68,  35,  35],
       [ 25,  16,  67,  66,  76]])
np.sqrt(a5)
array([[ 6.40312424, 10.        ,  5.09901951,  4.69041576,  7.14142843],
       [ 7.61577311,  6.8556546 ,  7.21110255,  7.07106781,  8.66025404],
       [ 4.79583152,  4.12310563,  2.44948974,  8.30662386,  6.244998  ],
       [ 5.83095189,  6.244998  ,  2.82842712,  6.92820323,  6.4807407 ],
       [ 4.69041576,  8.66025404,  8.24621125,  5.91607978,  5.91607978],
       [ 5.        ,  4.        ,  8.18535277,  8.1240384 ,  8.71779789]])
np.exp(a5)
array([[6.39843494e+17, 2.68811714e+43, 1.95729609e+11, 3.58491285e+09,
        1.40934908e+22],
       [1.54553894e+25, 2.58131289e+20, 3.83100800e+22, 5.18470553e+21,
        3.73324200e+32],
       [9.74480345e+09, 2.41549528e+07, 4.03428793e+02, 9.25378173e+29,
        8.65934004e+16],
       [5.83461743e+14, 8.65934004e+16, 2.98095799e+03, 7.01673591e+20,
        1.73927494e+18],
       [3.58491285e+09, 3.73324200e+32, 3.40427605e+29, 1.58601345e+15,
        1.58601345e+15],
       [7.20048993e+10, 8.88611052e+06, 1.25236317e+29, 4.60718663e+28,
        1.01480039e+33]])
np.log10(a5)
array([[1.61278386, 2.        , 1.41497335, 1.34242268, 1.70757018],
       [1.76342799, 1.67209786, 1.71600334, 1.69897   , 1.87506126],
       [1.36172784, 1.23044892, 0.77815125, 1.83884909, 1.59106461],
       [1.53147892, 1.59106461, 0.90308999, 1.68124124, 1.62324929],
       [1.34242268, 1.87506126, 1.83250891, 1.54406804, 1.54406804],
       [1.39794001, 1.20411998, 1.8260748 , 1.81954394, 1.88081359]])
list(range(0,10 , 2))
[0, 2, 4, 6, 8]
#Arange same as range but it also takes jump size as float
print(np.arange(10))
print(np.arange(1.8,10.7,2.5)) 
[0 1 2 3 4 5 6 7 8 9]
[1.8 4.3 6.8 9.3]
np.arange(1.8,10.7,2.5)
array([1.8, 4.3, 6.8, 9.3])
np.linspace(2,3,num=50,retstep=True)
(array([2.        , 2.02040816, 2.04081633, 2.06122449, 2.08163265,
        2.10204082, 2.12244898, 2.14285714, 2.16326531, 2.18367347,
        2.20408163, 2.2244898 , 2.24489796, 2.26530612, 2.28571429,
        2.30612245, 2.32653061, 2.34693878, 2.36734694, 2.3877551 ,
        2.40816327, 2.42857143, 2.44897959, 2.46938776, 2.48979592,
        2.51020408, 2.53061224, 2.55102041, 2.57142857, 2.59183673,
        2.6122449 , 2.63265306, 2.65306122, 2.67346939, 2.69387755,
        2.71428571, 2.73469388, 2.75510204, 2.7755102 , 2.79591837,
        2.81632653, 2.83673469, 2.85714286, 2.87755102, 2.89795918,
        2.91836735, 2.93877551, 2.95918367, 2.97959184, 3.        ]),
 0.02040816326530612)
#In between certain range, we want to generate numbers
print(np.linspace(2,3,num=50)) #Considers Linear Space
print(np.linspace(2,3,num=50,retstep=True)) #Jumpsize will also be returned
[2.         2.02040816 2.04081633 2.06122449 2.08163265 2.10204082
 2.12244898 2.14285714 2.16326531 2.18367347 2.20408163 2.2244898
 2.24489796 2.26530612 2.28571429 2.30612245 2.32653061 2.34693878
 2.36734694 2.3877551  2.40816327 2.42857143 2.44897959 2.46938776
 2.48979592 2.51020408 2.53061224 2.55102041 2.57142857 2.59183673
 2.6122449  2.63265306 2.65306122 2.67346939 2.69387755 2.71428571
 2.73469388 2.75510204 2.7755102  2.79591837 2.81632653 2.83673469
 2.85714286 2.87755102 2.89795918 2.91836735 2.93877551 2.95918367
 2.97959184 3.        ]
(array([2.        , 2.02040816, 2.04081633, 2.06122449, 2.08163265,
       2.10204082, 2.12244898, 2.14285714, 2.16326531, 2.18367347,
       2.20408163, 2.2244898 , 2.24489796, 2.26530612, 2.28571429,
       2.30612245, 2.32653061, 2.34693878, 2.36734694, 2.3877551 ,
       2.40816327, 2.42857143, 2.44897959, 2.46938776, 2.48979592,
       2.51020408, 2.53061224, 2.55102041, 2.57142857, 2.59183673,
       2.6122449 , 2.63265306, 2.65306122, 2.67346939, 2.69387755,
       2.71428571, 2.73469388, 2.75510204, 2.7755102 , 2.79591837,
       2.81632653, 2.83673469, 2.85714286, 2.87755102, 2.89795918,
       2.91836735, 2.93877551, 2.95918367, 2.97959184, 3.        ]), 0.02040816326530612)
np.logspace(2,4,num=4,base=10) #Considers Logarithmic Base
array([  100.        ,   464.15888336,  2154.43469003, 10000.        ])
#Identity Matrix
np.eye(5)
array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])
 
