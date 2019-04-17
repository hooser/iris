from data_proc import data_process
#import math
import numpy as np 

def h_theta(X,w):                       # w -> 5*1  X -> 90 * 5        
	temp_res = X * w                    # temp_res -> 90 * 1
	temp_res = np.mat(temp_res)
	#print('temp_res.shape = ',temp_res.shape)
	res = 1 / (1 + np.exp(-temp_res))
	#print('res.shape = ',res.shape)
	return res

def train():
	X1,y1,X2,y2,X3,y3 = data_process()  #读取相关矩阵
	sp1 = X1.shape                      #sp1为训练集形状
	m1 = sp1[0]                         #m为训练集的数目
	n1 = sp1[1]                         #n为训练集的维度
	y_l = len(y1)                       #y_l为训练集元组的数目
	wt = np.random.rand(3,n1+1)          #初始化权重矩阵, wt -> 3 * 5
	X1 = np.insert(X1,0,1,axis = 1)     #X1为增广矩阵
	#print(X1.shape)      ---> X1.shape = 90 * 5

	### 训练相关的参数
	iteration = 100000                    #学习次数
	K = 3                               #分类种类数目
	alpha = 0.1                         #学习速率
	lamb = 1                          #正则化参数

	### 转化为二分类
	y1 = np.mat(y1)
	y1 = np.transpose(y1)
	Yk = np.zeros((y_l,3))
	Yk[:,[0]] = y1 == 0
	Yk[:,[1]] = y1 == 1
	Yk[:,[2]] = y1 == 2

	for i in range(K):
		for j in range(iteration):
			temp = np.mat(wt[[i],:])
			w_temp = np.transpose(temp)           #w_temp ---> 5 * 1
			#w_temp = np.mat(w_temp)
			res = h_theta(X1,w_temp)              #res ---> 90 * 1
			temp_Yk = np.mat(Yk[:,[i]])
			#temp_Yk = np.transpose(temp_Yk)       #temp_Yk ---> 1 * 90
			delta = res - temp_Yk
			wt[[i],[0]] = wt[[i],[0]] - alpha * 1 / m1 * np.sum(delta)
			Xt = np.transpose(X1[:,1:])
			damit = np.transpose(Xt * delta)
			wt[[i],1:] = wt[[i],1:] - alpha * 1 / m1 * (damit + lamb * wt[[i],1:])
    
	### 处理测试集数据
	X3 = np.insert(X3,0,1,axis = 1)     #X3为增广矩阵
	y3 = np.mat(y3)
	y3 = np.transpose(y3)               # 30 * 1
	#print('wt.shape = ',wt.shape)       # 3 * 5
 
	return wt,X3,y3

if __name__ == '__main__':
	weight_,Xtest,ytest = train()
	print('weight_ = ',weight_)
	print('Xtest = ',Xtest.shape)
	print('ytest = ',ytest.shape)
	

	


