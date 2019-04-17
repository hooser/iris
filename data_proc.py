from sklearn import preprocessing
import csv
import numpy as np 

def data_process():
	file = []
	with open('iris.csv') as cf:
		c_r = csv.reader(cf)
		next(c_r)
		for row in c_r:
			file.append(row)

	#数据集乱序处理->file_n
	file_n = np.array(file)
	np.random.shuffle(file_n)
    
	data = []
	species = []                     # 种类(字符串)
	label = []                       # 种类(数值)
	for line in file_n:
		lineArr = []
		for i in range(4):
			lineArr.append(np.float32(line[i]))
		data.append(lineArr)
		species.append(line[-1])

	# 鸢尾花共有setosa,versiclor,virginica三类，将其类别转换为数字标签
	le = preprocessing.LabelEncoder()
	le.fit(["setosa", "versicolor", "virginica"])
	label = le.transform(species)
	data = np.array(data)

	#数据集归一化处理
	d_m = data.mean(axis = 0)
	d_s = data.std(axis = 0)
	data = (data - d_m) / d_s


	###构建训练集、cv集、测试集
	#训练集
	ytrain = []
	ycv = []
	ytest = []

	Xtrain = data[:90,:]
	ytrain = label[:90]
	#cv集
	Xcv = data[90:120,:]
	ycv = label[90:120]
	#测试集
	Xtest = data[120:,:]
	ytest = label[120:]

	return Xtrain,ytrain,Xcv,ycv,Xtest,ytest

if __name__ == '__main__':
	X1,y1,X2,y2,X3,y3 = data_process()
	Yk = np.zeros((len(y1),3))
