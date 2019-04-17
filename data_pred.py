from data_train import train 
from numpy import *

def predict():
	wt = mat(zeros((3,5)))
	Xtest = mat(zeros((30,5)))
	ytest = mat(zeros((30,1)))
	wt,Xtest,ytest = train()
	
	### 计算乘积
	mul = mat(zeros((30,3)))
	wt_t = transpose(wt)
	mul = dot(Xtest,wt_t)   # ?尚未搞懂 ---> mat形式的矩阵乘法需要使用dot方法
	
    ### 取较大的索引为种类
	res = mul.argmax(axis = 1)
	#print('res = ',res)

	count = 0
	for i in range(len(ytest)):
		if res[i] != ytest[i]:
			count = count + 1

	print('测试集共有30个样本，分类错误的样本数目为：',count)

if __name__ == '__main__':
	predict()