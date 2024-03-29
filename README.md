# iris

一、简介
-----

    学习机器学习有一段时间了，由于以前使用的是matlab，所以想使用python来实现一些机器学习的问题。
    鸢尾花分类问题时一个很经典的问题，我就想从这个问题入手吧。网上有一些相关的代码，但是我看到的几
    个都有些肉眼可见的缺陷，所以，我索性把网上的参考抛开，按照自己的思路实现一个。 会有不少缺陷，求
    大神轻喷 :) 
    
    
二、iris数据集
-----
    Iris（鸢尾花）数据集是多重变量分析的数据集。 数据集包含150行数据，分为3类，每类50行数据。 每行数
    据包括4个属性：Sepal Length（花萼长度）、Sepal Width（花萼宽度）、Petal Length（花瓣长度）、
    Petal Width（花瓣宽度）。可通过这4个属性预测鸢尾花属于3个种类的哪一类。所以本项目是利用lr方法进行
    多分类处理。引用Iris数据集的方法主要有在sklearn的sklearn库中导入iris数据集和下载官方的iris.csv文
    件。本项目使用后一种获取数据的方法。


三、数据预处理
-------
    将数据集进行数据类型的转换，将Sepal Length（花萼长度）、Sepal Width（花萼宽度）、Petal Length（花瓣长度）、
    Petal Width（花瓣宽度）等属性由string类型转换为float型。将species（种类）按照原来的三种顺序分别为{1,2,3}。
    因为原始数据集是按照相同种类紧密排列的方式组织的，不利于操作，所以需要将数据集打乱。然后对特征进行归一化处理。
    随后将数据集按照训练集、cv集、测试集6:2:2的比例进行划分。读入csv文件主要有使用python I/O写入和读取CSV文件、
    使用Pandas读取CSV文件、使用Tensorflow读取CSV文件等，本项目使用了第一种。计划使用cv集与训练集观察在不同学习率
    alpha下的learning curve变化，从而选择合适的alpha值。为了避免由于不同属性的数量级差距过大导致数量级较小的属性
    在训练过程中被忽略，所以对所有的x矩阵进行归一化处理。


四、训练思想
-----
    由于分类数目K=3，特征数目M=5（包括bias），所以随机设置权值矩阵weights 3*5。
    对于每一类类别，使用二分类的思想。即，若训练识别第i类的iris，对训练集进行如下操作：species（i）= species(i)==i，
    从而将训练集的结果分为0,1两类。然后，按照逻辑回归的算法在训练集上实现代码。为了避免过拟合，使用正则化的方法，正则
    化的参数为lambda，根据经验公式，取lambda = 1
    
五、预测思想
--------
    对于测试集Xtest，计算weigths*Xt，选取使得结果最大的索引i(i = 0,1,2)，作为样本所分的类别。
    
六、结果分析
--------
    最终选取学习率alpha = 0.1。 通过在一定范围内增加训练的迭代次数，可以分析分类错误样本数目明显下降。（总数为30）
    iteration = 1000，误分样本数为5
    iteration = 5000，误分样本数为3
    iteration = 10000，误分样本数为2
    
七、后续可能更新方面
------
    由于最近比较忙，后面有时间会对对训练可视化方面进行更新
