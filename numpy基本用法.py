# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:08:59 2020

@author: lx
"""
import numpy as np
#################数据的读取与导出######################################
#'npy'文件#
arr=np.arange(0,10)          #生成数组
np.save('some_array.npy',arr)#导出数据
np.load('some_array.npy')    #导入数据

#'npz'文件#
np.savez('array_archive.npz',a=arr,b=arr)      #导出数据(在未压缩文件中保存多个数组)
np.savez_compressed('arrays_compressed.npz',a=arr,b=arr)#(在压缩文件中保存多个数组)
arch=np.load('array_archive.npz')              #导入数据
arch['b']                                      #索引某个数组
################数组的生成与转换#######################################
#生成随机数组 (np.random.randn())
data=np.random.randn(2,3)
data
data*10
data+data
data.shape #查看数组各阶维数
data.dtype #查看数组的数据类型

##生成特定数组##
range(3)
np.arange(3) #元素依次生成,range()的数组版本
np.arange(6).reshape(2,3)

np.zeros(10) #生成元素全为0的数组
np.zeros((2,3))

np.empty((2,3)) #生成一个没有初始化数值的数组

##转换成数组##
data1=[1,2.5,6,0,1]
arr1=np.array(data1)
arr1
arr1.dtype
data2=[[1,2],[2,3]]
arr2=np.array(data2) #自动生成高阶数组
arr2
arr2.dtype
arr2.ndim #查看阶数
arr2.shape#查看各阶维数

###################数组的数据类型及其转换################################
arr=np.zeros(5,dtype=np.float) #生成时指定数据类型
arr
arr1=np.array([1,2,3],dtype=np.float64) #转换时指定数据类型
arr2=np.array([1,2,3],dtype=np.int32)
arr1.dtype #查看数据类型
arr2.dtype

arr=np.array([1,2,3,4,5])
arr.dtype
float_arr=arr.astype(np.float) #转换数组的数据类型
float_arr.dtype

arr=np.array([1.2,2.5,3.0])
arr
arr.astype(np.int32) #将浮点型转换成整型时,结果直接去掉小数点

numeric_strings=np.array(['1.2','2.3'],dtype=np.string_)
numeric_strings.dtype
numeric_strings.astype(float) #字符串变浮点型

int_array=np.arange(10)
a=np.array([1.2,3.2],dtype=np.float64)
int_array.astype(a.dtype) #转换时使用另一个数组的数据类型

empty_uin32=np.empty(8,dtype='u4')
empty_uin32 #生成数组时使用类型代码

############################ 数组基本运算 ##################################
arr=np.array([[1.,2.,3.],[4.,5.,6.]])
arr

arr*arr #各元素相乘

arr-arr #各元素相减

arr**0.5#数乘

1/arr

arr2=np.array([[0.,4.,1.],[7.,2.,12.]])
arr2

arr2>arr #数组比较，产生bool阵

############################ 数组的索引 ##################################
##第一种 切片索引(非复制)##
arr=np.arange(10)
arr[1:6] #第二个至第六个

arr2=np.arange(9).reshape(3,3)
arr2

arr2[:2] #前两行

arr2[:2,1:] #前两行且第一列之后

arr2[1,:2] #第二行且前两列

arr2[:,:1]
#对比(前者三行一列,后者一行三列)
arr2[:,0]
#对比
arr2[:,0:1] #(三行一列)

arr2[1,:2]=1000 #给切片赋值后原数组会改变（意味着切片非复制）
arr2            #显然改变
#甚至#
a=arr2[1,:2]
a[:2]=1000000
arr2

##第二种 布尔索引(复制)##
title=np.array(['1','2','3'])
data=np.random.randn(3,4)
title
data
title=='2'
data[title=='2'] #传入bool矩阵索引

data[title=='2',:2] #进一步选择列

data[title!='1']
#等价于
data[~(title=='1')]

data[(title=='1')|(title=='2')] #'或'的运用

data[data<0]=0     #例子1说明布尔阵为复制
data
#但是
a=data[data<0]
a[1]=10000
a
data          #原数组未发生变化,说明此索引是复制

data[title=='1']=10000 #例子2说明布尔阵为复制
data
#但是
b=data[title=='1']
b[0:]=10000
b             
data#原数组未发生变化,说明此索引是复制

##第三种 神奇索引(复制)##
arr=np.empty((8,4))
for i in range(8):
    arr[i]=i
arr

arr[[4,3,0,6]] #选对应行数

arr[[-3,-1,-2]] #倒着选

arr=np.arange(32).reshape((8,4))
arr[[1,2],[2,3]] #离散选点,结果为一维
arr[[1,2]][:,[1,3]] #离散选点,结果为二维

c=arr[[4,3,0,6]]
c[1]=100000
c
arr         #原数组未发生变化,说明此索引是复制

############################ 数组的转置和换轴 ##################################
arr=np.arange(15).reshape((3,5))
arr
arr.T #转置

np.dot(arr.T,arr) #矩阵乘法

arr=np.arange(16).reshape((2,2,4))
arr
arr.transpose((1,0,2)) #transpose换轴

arr.swapaxes(1,2) #swapaxes换轴(未复制)

#################### 通用函数(ufuc)： 快速的逐元素数组函数 ##################################
arr=np.arange(10)
arr

np.sqrt(arr) #逐个开平方根

np.exp(arr) #逐个取指数

x=np.random.randn(8)
y=np.random.randn(8)
np.maximum(x,y) #逐个求最大值

arr=np.random.randn(7)*5
arr
remainder,whole_part=np.modf(arr) #逐个输出整数和小数
remainder
whole_part 

arr
np.sqrt(arr) 
np.sqrt(arr,arr) #通用函数可将结果直接赋给arr
arr  

########################## 面向数组编程 #########################################
##小应用:函数绘图##
points=np.arange(-5,5,0.01) #生成1000个点
xs,ys=np.meshgrid(points,points)
xs
ys
z=np.sqrt(xs**2+ys**2) #使用向量化函数,避免for循环！
z

import matplotlib.pyplot as plt #将函数可视化
plt.imshow(z,cmap=plt.cm.gray);plt.colorbar()
plt.title("$\sqrt{x^2+y^2}$")

##将条件逻辑作为数组操作##
xarr=np.array([1.1,1.2,1.3,1.4,1.5])
yarr=np.array([2.1,2.2,2.3,2.4,2.5])
cond=np.array([True,False,True,True,False])

result=[(x if c else y) for x,y,c in zip(xarr,yarr,cond)] #传统列表循环法
result

result=np.where(cond,xarr,yarr) #np.where法
result

arr=np.random.randn(4,4)
arr
arr>0
np.where(arr>0,2,-2) #运用np.where做替换操作

np.where(arr>0,2)     #语法错误
np.where(arr>0,2,arr) #只替换大于零的(原数据未改变,括号中'arr'并非赋值意思)

##统计量函数##
arr=np.random.randn(5,4)
arr

arr.mean()  #求均值
#等价于#
np.mean(arr)

arr.sum()   #求和

arr.mean(axis=1) #计算每一行均值
#等价于#
arr.mean(1) 

arr.sum(axis=0) #计算每一列和
#等价于#
arr.sum(0)

arr=np.array([0,1,2,3,4,5,6,7])
arr.cumsum() #累积和

arr=np.array([[0,1,2],[3,4,5],[6,7,8]])
arr
arr.cumsum(axis=0) #列累积和
arr.cumprod(axis=1)#行累积积

##布尔值数组的方法##
arr=np.random.randn(1,100)
(arr>0).sum()      #查看大于0元素的个数

bools=np.array([False,False,True])
bools.any() #是否至少有一个为True
bools.all() #是否全为Ture

##排序##
arr=np.random.randn(1,6)
arr
arr.sort()

arr=np.random.randn(5,3)
arr
arr.sort(1) #按轴排序
arr

large_arr=np.random.randn(1000)
large_arr.sort() #分位数
large_arr[int(0.05*len(large_arr))]                     

##唯一值与其他集合逻辑##
names=np.array(['Bob','Joe','Will','Joe','Joe'])
np.unique(names) #唯一值排序

ints=np.array([3,3,3,2,2,1,1,4,4])
np.unique(ints)
#与纯python法比较#
sorted(set(ints))

values=np.array([6,0,0,3,2,5,6])
np.in1d(values,[2,3,6,7]) #查看一个数组的元素是否在另一个数组中

np.intersect1d(np.array([1,2,3]),[2,3,4]) #求交集
#等价于#
np.intersect1d([1,2,3],[2,3,4])

################################ 线性代数 ####################################
x=np.array([[1.,2.,3.],[4.,5.,6.]])
y=np.array([[6.,23.],[-1,7],[8,9]])
x
y
x*y      #逐个元素相乘

x.dot(y) #矩阵乘法
#等价于#
np.dot(x,y)
#等价于#
x@y

##矩阵分解##
from numpy.linalg import inv,qr,svd
X=np.random.randn(5,5)
mat=X.T.dot(X)
inv(mat)           #求逆
mat.dot(inv(mat))
q,r=qr(mat) #矩阵的QR分解
q           #正交矩阵
r           #上三角矩阵
a,b,c=svd(mat) #奇异值分解

########################### 伪随机数生成 ##################################
samples=np.random.normal(size=(4,4))
samples

np.random.seed(1234)  #更改随机数种子
rng=np.random.RandomState(1234)
rng.randn(10)

np.random.randint(0,3)             #根据给定的由低到高的范围随机生成整数
np.random.uniform(-1,0,size=(1,10))#均匀[-1,0)分布
np.random.rand(10) #生成来自"标准"均匀分布[0,1)的十个观测
np.random.normal(0,1,size=(1,10))   #正态分布
np.random.randn(10) #生成来自标准正态分布的十个观测
np.random.binomial(1,0,20) #二项分布
np.random.beta(10)     #beta分布
np.random.chisquare(10)#卡方分布
np.random.gamma(10)   #伽马分布

########################### 例子：随机游走 #####################################
##随机游走的可视化##
import random
import matplotlib.pyplot as plt
position=0                                #实现1000步的随机漫步
walk=[position]
steps=1000
for i in range(steps):
    step=1 if random.randint(0,1) else -1
    position += step
    walk.append(position)
plt.plot(walk[:100])                      #前100步的可视化

##提取统计数据##
nsteps=1000
draws=np.random.randint(0,2,size=nsteps)
steps=np.where(draws>0,1,-1)
walk=steps.cumsum() #累积求和
walk.min()          #最左位置
walk.max()          #最右位置
(np.abs(walk)>=10).argmax() #返回第一次到达位置10或-10的时刻

##一次模拟多次随机漫步##
nwalks=5000
nsteps=1000
draws=np.random.randint(0,2,size=(nwalks,nsteps)) #0或1
steps=np.where(draws>0,1,-1)
walks=steps.cumsum(1)
walks.max()
walk.min()
hists30=(np.abs(walks)>=30).any(1)
hists30
hists30.sum() #首次达到30或-30的时刻

steps=np.random.normal(loc=0,scale=0.25,size=(nwalks,nsteps)) #基于正态分布的随机游走
steps